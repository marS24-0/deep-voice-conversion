import os
from inference import *

file_list = []
device = None

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
def f0_extract(item_index):
    """
    extracts mean and std from (original and log-scaled) f0, and saves these 4 values in an npy file with the audio length (discarding zeros of F0)
    """
    x, gt_audio, _, filename = dataset[item_index] # retrieve audio item
    f0 = x['f0'] # shape (1, 1, length)
    f0n = f0.numpy()[0][0] # remove dimensions
    mask = ~np.isclose(f0n, 0.0) # discard 0 values
    values = f0n[mask]
    logvalues = np.log(values) # compute logarithm of non-zero values
    f0c = np.array((values.size, values.mean(), logvalues.mean(), values.std(), logvalues.std()), dtype=np.float32) # save mean and std, for both original and log-scaled values, and length of non-zero F0
    np.save(f'{a.output_dir}/{filename}.npy', f0c)



def main():
    global file_list
    global device
    print('Initializing f0 Extraction Process..')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_wav_dir', type=Path, default='xvector_test/f0/input_wavs', help='directory with audios to read, to use in alternative to --input_test_file')
    parser.add_argument('--output_dir', default='xvector_test/f0/output_f0s', help='directory where to save extracted F0s')
    parser.add_argument('--input_test_file', type=Path, default=None, help='list of audio files to read')
    parser.add_argument('--test_wav_dir', default=None, help='ignored')
    parser.add_argument('--feat_model', type=Path, help='ignored')
    parser.add_argument('--kmeans_model', type=Path,nargs="?",default=None, help='ignored')
    parser.add_argument('--soft_model', type=Path,nargs="?",default=None, help='ignored')
    parser.add_argument('--checkpoint_file', default='pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze', help='gan model parameters')
    parser.add_argument('--f0_dir', type=Path, help='ignored')
    parser.add_argument('--xv_dir', type=Path, help='ignored')
    parser.add_argument('--cpu', default=False, action='store_true', help='cpu-only')

    a = parser.parse_args()

    device = init_torch(a.cpu)

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

    if a.input_test_file is not None:
        file_list = []
        for line in open(a.input_test_file):
            file_list.append(line.strip())
    else:
        file_list = [f'{a.source_wav_dir}/{file}'for file in os.listdir(a.source_wav_dir)]

    init_worker(a)

    for i in range(0, len(dataset)):
        f0_extract(i)
        bar = progbar(i, len(dataset))
        message = f'{bar} {i}/{len(dataset)} '
        stream(message)



if __name__ == "__main__":
    main()
    print()
