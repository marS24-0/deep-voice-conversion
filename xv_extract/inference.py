# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# This source code was adapted from https://github.com/facebookresearch/speech-resynthesis by Xiaoxiao Miao (NII, Japan).
from os import listdir, getcwd
from os.path import isfile, join, isdir
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
from readwrite import write_raw_mat

h = None
device = None

def init_torch(cpu=False):
    global device
    device = torch.device("cuda" if not cpu and torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))
    return device

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
    # no generation, only computes and return the xvector
    # see latentGenerator.gen_vpc in models_test.py
    return generator.gen_vpc(xv_path, **x).to(device)

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

    generator = latentGenerator(h, device).to(device)
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
def inference(item_index):
    # no inference, only computes and returns the xvector
    x, gt_audio, _, filename = dataset[item_index]
    x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}
    gt_audio = torch.autograd.Variable(gt_audio.to(device, non_blocking=False))
    fname_out_name = Path(filename).stem
    xv_path = str(a.xv_dir) + '/' + fname_out_name + '.xvector'
    xv = generate(h, generator, x, xv_path)
    write_raw_mat(xv.cpu().numpy(), xv_path)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, default=None, help="[single wav file | dir of wav files | lst file]")
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--xv_dir', required=True, type=Path, help="output dir where to save the extracted xvectors")
    parser.add_argument('--cpu', default=False, action='store_true')

    a = parser.parse_args()

    init_torch(a.cpu)

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
