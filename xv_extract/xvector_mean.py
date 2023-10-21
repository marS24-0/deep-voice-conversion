import argparse
import os
import numpy as np
from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from readwrite import read_raw_mat, write_raw_mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, required=True, help='input dir containing all the xvectors')
    parser.add_argument('--output', default=None, help='output dir (input dir is default)')
    parser.add_argument('--dataset', default='libri', choices=['libri', 'vctk'])
    a = parser.parse_args()

    # read files and group by speaker_id
    d = defaultdict(lambda: [])
    for file in os.listdir(a.input):
        filename = os.fsdecode(file)
        if filename.endswith(".xvector"):
            speaker_id = filename.split('-')[0] if a.dataset=='libri' else filename.split('_')[0]
            xv = read_raw_mat(a.input + "/" + filename, 192)
            d[speaker_id].append(xv)
    
    # average
    for k, v in d.items():
        mean = np.array(v).mean(0)
        if a.output is None:
            write_raw_mat(mean, a.input + '/' + k + ".xvector")
        else:
            write_raw_mat(mean, a.output + '/' + k + ".xvector")
        
if __name__ == '__main__':
    main()
