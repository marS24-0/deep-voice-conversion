# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This source code was adapted from https://github.com/facebookresearch/speech-resynthesis by Xiaoxiao Miao (NII, Japan).


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')
import random
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset_test import latentDataset, mel_spectrogram, get_dataset_filelist
from models_test import latentGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, \
    save_checkpoint, build_env, AttrDict
from scipy import signal
torch.backends.cudnn.benchmark = True
from torch.nn import CosineEmbeddingLoss

from scripts.readwrite import read_raw_mat
import numpy as np


def train(rank, local_rank, a, h):
    if h.num_gpus > 1:
        print(h.num_gpus)
        print(f"MASTER_ADDR: ${os.environ['MASTER_ADDR']}")
        print(f"MASTER_PORT: ${os.environ['MASTER_PORT']}")
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            rank=rank,
            world_size=h.num_gpus,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(local_rank))

    # read target data
    xv_target = read_raw_mat(a.xv_path, 192)
    xv_target = torch.FloatTensor(xv_target).unsqueeze(0)
    xv_shape = (h.batch_size,) + xv_target.shape[1:]
    xv_target = torch.broadcast_to(xv_target, xv_shape).to(device)
    f0_target = np.load(a.f0_path)
    

    generator = latentGenerator(h, xv_target=xv_target, f0_target=f0_target).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(
            generator,
            device_ids=[local_rank],find_unused_parameters=True
        ).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[local_rank],find_unused_parameters=True).to(device)
        msd = DistributedDataParallel(msd, device_ids=[local_rank],find_unused_parameters=True).to(device)
        print("dataparallel")

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = latentDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
                           device=device,f0=h.get('f0', None))


    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.get("num_workers",0), shuffle=False, sampler=train_sampler,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validset = latentDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                                   h.hop_size,
                                   h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                                   fmax_loss=h.fmax_for_loss,
                                   device=device, f0=h.get('f0', None))

        validation_loader = DataLoader(validset, num_workers=h.get("num_workers",0), shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    if not h.get('hifigan_freeze',None):
        mpd.train()
        msd.train()
    print(f'debug last_epoch {last_epoch}\n', end='')

    for epoch in range(max(0, last_epoch), a.training_epochs + max(0, last_epoch)):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
    
        
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            x, y, y_mel,_ = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=False))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
            y = y.unsqueeze(1)
            x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}

            # # extract source xvector
            # audio_data = x['audio']
            # with torch.no_grad():
            #     xv_source = generator.fbank(audio_data.squeeze(1))
            #     xv_source = generator.mean_var_norm(xv_source, torch.ones(xv_source.shape[0]).to(xv_source.device))
            # xv_source, _ = generator.xv_model(xv_source)
            # xv_source = F.layer_norm(xv_source, xv_source.shape)
            # xv_source = xv_source.transpose(2, 1)

            y_g_hat = generator(**x)


            if generator.xv_feature == 'fbank':
                # fbank input (batchsize,wav_len)
                # with torch.no_grad():
                xv_input = generator.fbank(y_g_hat.squeeze(1))
                xv_input = generator.mean_var_norm(xv_input, torch.ones(xv_input.shape[0]).to(xv_input.device))
                xv_gen, _ = generator.xv_model(xv_input)
            elif generator.xv_feature == 'ssl':
                # ssl_xv_model input (batchsize,wav_len), output xv_input (batchsize,frames,768)
                xv_input = generator.ssl_xv_model(y_g_hat.squeeze(1))
                xv_gen, _ = generator.xv_model(xv_input)

            # extract generated xvector
            # xv_gen = generator.xv_model(y_g_hat)
            xv_gen = F.layer_norm(xv_gen, xv_gen.shape)
            xv_gen = xv_gen.transpose(2, 1)

            # cosine similarity loss
            """
            torch.nn.CosineEmbeddingLoss
            Creates a criterion that measures the loss given input tensors x1x1​, x2x2​ and a Tensor label yy with values 1 or -1.
            This is used for measuring whether two inputs are similar or dissimilar, using the cosine similarity, and is typically
            used for learning nonlinear embeddings or semi-supervised learning.
            
            ref: https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
            """

            print(xv_target.shape, xv_gen.shape)
            cos_criterion = CosineEmbeddingLoss()
            loss_cosine = cos_criterion(xv_target.transpose(2, 1).squeeze(), xv_gen.squeeze(), torch.ones(h.batch_size, requires_grad=False, device=device)) # TODO: check the correctness

            assert y_g_hat.shape == y.shape, f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"
            #print("train_vc line148",y_g_hat.shape)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size, h.fmin, h.fmax_for_loss)
            

            optim_d.zero_grad()

            # MPD

            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())

            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + 100*loss_cosine # cosine loss added

            loss_gen_all.backward()

            for p in generator.xv_model.parameters():
                p.grad = None

            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print(
                        'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(steps,
                                                                                                                  loss_gen_all,
                                                                                                                  mel_error,
                                                                                                                  time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                                      'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                                                      'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                                      'steps': steps, 'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    print(f'debug loss_cosine {loss_cosine}')
                    sw.add_scalar("training/loss_cosine", loss_cosine, steps)
                    if h.get('f0_vq_params', None):
                        sw.add_scalar("training/commit_error", f0_commit_loss, steps)
                        sw.add_scalar("training/used_curr", f0_metrics['used_curr'].item(), steps)
                        sw.add_scalar("training/entropy", f0_metrics['entropy'].item(), steps)
                        sw.add_scalar("training/usage", f0_metrics['usage'].item(), steps)
                    if h.get('code_vq_params', None):
                        sw.add_scalar("training/code_commit_error", code_commit_loss, steps)
                        sw.add_scalar("training/code_used_curr", code_metrics['used_curr'].item(), steps)
                        sw.add_scalar("training/code_entropy", code_metrics['entropy'].item(), steps)
                        sw.add_scalar("training/code_usage", code_metrics['usage'].item(), steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, y_mel,_= batch
                            x = {k: v.to(device, non_blocking=False) for k, v in x.items()}

                            y_g_hat = generator(**x)
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
                            #print("train_vc line248",y_g_hat.shape)
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel[0].cpu()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat[:1].squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec[:1].squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    generator.train()
            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    if rank == 0:
        print('Finished training')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--training_steps', default=400000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=10000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed-world-size', type=int)
    parser.add_argument('--distributed-port', type=int)
    parser.add_argument('--xv_path', required=True)
    parser.add_argument('--f0_path', required=True)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available() and 'WORLD_SIZE' in os.environ:
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = int(os.environ['WORLD_SIZE'])
        h.batch_size = int(h.batch_size / h.num_gpus)
        local_rank = a.local_rank
        rank = a.local_rank
        print('Batch size per GPU :', h.batch_size)
    else:
        rank = 0
        local_rank = 0

    train(rank, local_rank, a, h)


if __name__ == '__main__':
    main()

