import numpy as np
from sklearn.metrics import det_curve
import json

def get_eer(y_true, y_score):
    fpr, fnr, thresholds = det_curve(y_true, y_score)
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, fpr, fnr, thresholds

ASV_PATH = "./ASV/"

datasets = ["libri", "vctk"]
partitions = ["dev", "test"]
trials = ["trials_m", "trials_f"]
gen = ["_gen_", "_"]

dictionary = {}

for g in gen:
    for d in datasets:
        for p in partitions:
            for t in trials:
                    path = ASV_PATH + d + "_" + p + "/det" + g + t if d == "libri" else ASV_PATH + d + "_" + p + "/det" + g + t + "_mic2"
                    filename = g[1:] + d + "_" + p + "_" + t
                    x = np.load(path + ".npz")
                    eer, _, _, _ = get_eer(x['y_true'], x['y_pred'])
                    dictionary[filename] = f"{eer*100:.3}"


with open("scores.json", "w") as f:
    json.dump(dictionary, f)
