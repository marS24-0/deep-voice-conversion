# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# This source code was adapted from https://github.com/facebookresearch/speech-resynthesis by Xiaoxiao Miao (NII, Japan).

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

from multiprocessing import Manager, Pool
import librosa
import numpy as np
import torch
from scipy.io.wavfile import write

from dataset_test import latentDataset, mel_spectrogram, \
    MAX_WAV_VALUE
from utils import AttrDict
from models_test import latentGenerator,SoftPredictor
import joblib
import fairseq
from os import listdir, getcwd
from os.path import isfile, join, isdir

h = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ' + str(device))


def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def generate(h, generator, x, xv_path):
    start = time.time()
    y_g_hat = generator.gen_vpc(xv_path,**x).to(device)
    if type(y_g_hat) is tuple:
        y_g_hat = y_g_hat[0]
    rtf = (time.time() - start) / (y_g_hat.shape[-1] / h.sampling_rate)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio, rtf


def init_worker(arguments):
    import logging
    logging.getLogger().handlers = []

    global generator
    global dataset
    global device
    global a
    global h

    a = arguments

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = latentGenerator(h, a.xv_path).to(device)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g['generator'])


    file_list = []
    if str(a.input).endswith("lst"):
        for line in open(a.input):
            file_list.append(line.strip())
    else:
        if isdir(str(a.input)):
            file_list = [join(str(a.input), f) for f in listdir(str(a.input)) if isfile(join(str(a.input), f))] 
        else:
            file_list = [str(a.input)]
    
    dataset = latentDataset(file_list, -1, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                              h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss,  device=device)

    os.makedirs(a.output_dir, exist_ok=True)


    generator.eval()
    generator.remove_weight_norm()

    # fix seed
    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def f0_replace(f0, filename): # NOTE: integrity of f0 is not preserved
    """
    replace F0
    inputs:
        f0: torch.tensor, source F0
        filename: target F0 file (basename without extension), as produced by f0_extract.py or f0_avg.py
    returns: torch.tensor, adapted F0
    """
    if a.no_f0:
        return f0
    target_f0_name = str(a.xv_path).replace('.xvector', '.npy')
    target = np.load(target_f0_name)
    f0_source = f0.cpu().numpy()[0][0] # remove dimensions
    mask = ~np.isclose(f0_source, 0.0) # discard 0 values
    values = f0_source[mask]
    if a.f0_log: # compute logarithm of source and consider log-scaled target
        values = np.log(values)
        mean = target[2]
        std = target[4]
    else: # consider original target
        mean = target[1]
        std = target[3]
    values -= values.mean() # subtract source mean
    if a.f0_std: # adapt standard deviation
        values *= std / values.std()
    values += mean # add target mean
    if a.f0_log: # exponentiate the adapted log-F0 to the original scale
        values = np.exp(values)
    f0_new = torch.tensor(values, device='cuda')
    f0[0,0,mask] = f0_new
    return f0

@torch.no_grad()
def inference(item_index):
    x, gt_audio, _, filename = dataset[item_index]
    x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}
    x['f0'] = f0_replace(x['f0'], filename)
    gt_audio = torch.autograd.Variable(gt_audio.to(device, non_blocking=False))
    fname_out_name = Path(filename).stem
    xv_path = str(a.xv_path)
    audio, rtf = generate(h, generator, x, xv_path)
    output_file = os.path.join(a.output_dir, fname_out_name + '_gen.wav')
    audio = librosa.util.normalize(audio.astype(np.float32))
    write(output_file, h.sampling_rate, audio)

    if a.gt_audio:
        output_file = os.path.join(a.output_dir, fname_out_name + '_gt.wav')
        gt_audio = librosa.util.normalize(gt_audio.squeeze().cpu().numpy().astype(np.float32))
        write(output_file, h.sampling_rate, gt_audio)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, default=None, help='list of audio files to read (.lst) OR directory with audios to read OR path of single audio to read, to convert')
    parser.add_argument('--gt_audio', action='store_true', help="create gt_audio")
    parser.set_defaults(gt_audio=False)
    parser.add_argument('--test_wav_dir', default=None, help='ignored')
    parser.add_argument('--feat_model', type=Path, help='ignored')
    parser.add_argument('--kmeans_model', type=Path,nargs="?",default=None, help='ignored')
    parser.add_argument('--soft_model', type=Path,nargs="?",default=None, help='ignored')
    parser.add_argument('--output_dir', default='output', help='directory where to save generated audios')
    parser.add_argument('--checkpoint_file', required=True, help='gan model parameters')
    parser.add_argument('--xv_path', type=Path, default=None, help='directory with target speaker vectors (as saved by xv_extract/ or xvector_mean.py)')
    parser.add_argument('--f0_std', default=False, action='store_true', help='adapt F0 standard deviation')
    parser.add_argument('--f0_log', default=False, action='store_true', help='convert F0 in logarithmic scale')
    parser.add_argument('--no_f0', default=False, action='store_true', help='do not adapt F0')

    a = parser.parse_args()

    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    if not os.path.isfile(cp_g) or not os.path.exists(cp_g):
        print(f"Didn't find checkpoints for {cp_g}")
        return

    file_list = []
    if str(a.input).endswith("lst"):
        for line in open(a.input):
            file_list.append(line.strip())
    else:
        if isdir(str(a.input)):
            file_list = [join(str(a.input), f) for f in listdir(str(a.input)) if isfile(join(str(a.input), f))] 
        else:
            file_list = [str(a.input)]

    init_worker(a)

    for i in range(0, len(dataset)):
        inference(i)
        bar = progbar(i, len(dataset))
        message = f'{bar} {i}/{len(dataset)} '
        stream(message)

if __name__ == '__main__':
    main()
    print()
