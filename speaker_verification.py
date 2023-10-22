DESC = """
This machinery runs speaker verification from existing protocols and pre-extracted xvectors from this repository.
Yes, I am implementing this from scratch, deal with it.
NOTE: I am using automatically adding _gen suffix in the case the pre-extracted vector is in npy format instead of raw matrix, because this is the only use case.
It's not great, but it is going to save a lot boilerplate.
"""

import pdb
from argparse import ArgumentParser
import os
import sys
sys.path.append('/medias/speech/projects/panariel/SSL-SAS-folder/SSL-SAS') # just give me visibility to everything lol

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from tqdm import tqdm
from sklearn.metrics import det_curve, roc_curve

from scripts.readwrite import read_raw_mat




parser = ArgumentParser(description=DESC)

parser.add_argument('--enrolls', type=str, required=True, help='Enrollment file, each row is an utterance ID and nothing else, speaker is inferred from ID.')
parser.add_argument('--trials', type=str, required=True, help='Trials file, each row has format <speaker id> <utterance id> "target"/"nontarget".')
parser.add_argument('--enrolls_root', type=str, required=True, help='Where to find the pre-extracted xvectors of the enrollment partition. They can either be in .npy format or in that raw matrix crap format used in this damn framework.')
parser.add_argument('--trials_root', type=str, required=True, help='Where to find the pre-extracted xvectors of the trials partition. They can either be in .npy format or in that raw matrix crap format used in this damn framework. Note that this has to be specified independently from the enrollment root since, of course, whoever came up with this codebase thought it made sense to export xvectors in separate folders. *sigh*')
parser.add_argument('--rates_file', default=None, help='Filename (with no extension) where to save the rates to plot the det. If not given, will not save anything.')
parser.add_argument('--dataset', default='libri', help='Which kind of dataset to use (influences the parsing approach). Defaults to "libri". For vctk, use... well, "vctk". Suprise!')


def speaker_from_uttid(uttid, dataset='libri'):
    if dataset == 'libri':
        return uttid.split('-')[0]
    elif dataset == 'vctk':
        return uttid.split('_')[0]
    else:
        raise ValueError("Dataset value shoudl be 'libri' or 'vctk'.")


def get_eer(y_true, y_score):
    """
    With ECAPA, we use cosine similarity. Thus, the convention is:
    - nontarget trials have label -1
    - target trials have label 1
    This is because cosine similarity is 1 when the vectors have identical orientation (i.e. angle 0)
    and it is -1 when they have completely opposite orientation (i.e. angle 180 degrees)

    This is my own re-implementation using det_curve function.
    The one I copypasted for millennia was using roc_curve, which really bothered me lol
    Also it's useful to visualize det curves now, so yeah
    """
    fpr, fnr, thresholds = det_curve(y_true, y_score)
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, fpr, fnr, thresholds

def get_eer_old(y_true, y_score):
    """
    This is the the old version of the function, just used as a double check.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    return eer

def read_xvector(root, uttid):
    """
    Try to read xvector in raw format.
    If not found, read it in npy format.
    If still not found, go f**k yourself.
    """
    try:
        fp = os.path.join(root, f'{uttid}.xvector')
        xvect = read_raw_mat(fp, 192)
    except FileNotFoundError as fnfe:
        # print(f'xvect raw of {uttid} not found, trying np')
        fp = os.path.join(root, f'{uttid}_gen.xvector.npy')
        xvect = np.load(fp)
    return xvect


if __name__ == "__main__":
    args = parser.parse_args()


    print(f'Constructing models from {args.enrolls}...')
    # Create speaker models from enrollz
    with open(args.enrolls, 'r') as f:
        enrolls_ids = [line.strip() for line in f.readlines()]

    
    # enrolls is usually small so let's just read everything at once :)))))
    models = {}
    for uttid in enrolls_ids:
        # fp = os.path.join(args.enrolls_root, f'{uttid}.xvector')
        # xvect = read_raw_mat(fp, 192)
        xvect = read_xvector(args.enrolls_root, uttid)

        spkid = speaker_from_uttid(uttid, dataset=args.dataset)
        #print(spkid)
        if spkid not in models:
            models[spkid] = [xvect]
        else:
            models[spkid].append(xvect)

    # Average utts to get models, each is of shape (192,)
    all_same_enroll = []
    for spkid, xvects in models.items():
        all_same = all([np.allclose(xvects[0], other) for other in xvects])
        all_same_enroll.append(all_same)
    if all(all_same_enroll):
        print("\tEnrollment vectors are all the same speakerwise.")

    models = {spkid: np.concatenate(xvects).mean(0) for spkid, xvects in models.items()}
    print('Done.')

    print(f'Computing trials from {args.trials}...')
    y_true = []
    y_pred = []
    # Read the scores one by one
    with open(args.trials, 'r') as f:
        trials_lines = [line.strip() for line in f.readlines()]
    
    
    for line in tqdm(trials_lines, desc='Producing trial scores'):
        spkid, utt_id, hr_label = line.split(' ')
        # select right label
        if hr_label == 'target':
            label = 1
        elif hr_label == 'nontarget':
            label = -1
        else:
            raise ValueError(f"Something went wrong: found label {hr_label} (should be 'target' or 'nontarget')")

        # read xvect and get appropriate speaker model
        # fp = os.path.join(args.trials_root, f'{utt_id}.xvector')
        # trial_xv = read_raw_mat(fp, 192)
        trial_xv = read_xvector(args.trials_root, utt_id)
        trial_xv = trial_xv.squeeze()

        model = models[spkid]
        
        # ready to produce the score
        # cosine_similarity = 1 - cosine_distance
        # (doing this because scipy has cos dist but not cos sim)
        score = 1 - cosine_distance(model, trial_xv)

        y_true.append(label)
        y_pred.append(score)

    # Compute eer and double check with the old function because better safe than sorry
    eer, fpr, fnr, thresholds = get_eer(y_true, y_pred)
    eer_old  = get_eer_old(y_true, y_pred)

    print('--- SCORES ---')
    print(f'EER: {eer*100:.3}')
    print(f'EER with old function: {eer_old*100:.3}')

    if args.rates_file is not None:
        print(f'Saving fpr and fnr to {args.rates_file}')
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        np.savez(args.rates_file, **{'fpr': fpr, 'fnr': fnr, 'y_true': y_true, 'y_pred': y_pred, 'thresh': thresholds})