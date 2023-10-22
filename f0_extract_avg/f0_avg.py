import argparse
import os
import numpy as np
from pathlib import Path

def avg(f0s:list) -> np.ndarray:
    """
    Average of F0s
    input: f0s, list of numpy arrays, each one containing length (of non-zero F0) of the related audio, means and deviations of original and log-scaled F0 (what is saved by f0_extract.py)
    output: numpy array, contains the weighted averages of means and deviations, with the sum of the lengths, in the same format of the input elements
    """
    v = np.array(f0s)
    w = v[:, 0] # weigths
    m = v[:, 1:3] # old means
    mm = np.average(m, axis=0, weights=w) # new mean (mean of old means)
    d = np.square(m-mm) # square of difference of means wrt means
    cov = np.average(d, axis=0, weights=w) # kind of covariance among sets
    s = np.square(v[:, 3:]) # old variances
    s = np.average(s, axis=0, weights=w) # average of old variances
    s += cov # new variances
    s = np.sqrt(s) # new stds
    w = np.array([w.sum()], dtype=np.float64) # new weight
    r = np.concatenate((w, mm, s))
    return r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default='xvector_test/f0/output_f0s', help='directory with extracted F0s to average')
    parser.add_argument('--output', type=Path, default='xvector_test/f0/average_f0.npy', help='output npy file')
    a = parser.parse_args()
    f0s = [np.load(f'{a.input}/{x}') for x in os.listdir(a.input)]
    f0 = avg(f0s)
    np.save(a.output, f0)

if __name__ == "__main__":
    main()
