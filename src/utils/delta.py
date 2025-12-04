#CODE COPIED AND ADJUSTED FROM https://github.com/htdt/hyp_metric/blob/master/delta.py

import torch
import torch.nn as nn
import torchvision
from scipy.spatial import distance_matrix
import numpy as np
from tqdm import tqdm
import sys


# def delta_hyp(dismat): # Takes in a distance matrix
#     p = 0
#     row = dismat[p, :][None, :]
#     col = dismat[:, p][:, None]
#     XY_p = 0.5 * (row + col - dismat)
#     maxmin = torch.minimum(XY_p[:, :, None], XY_p[None, :, :]).max(1).values
#     return (maxmin - XY_p).max()


def calc_hyperbolicity(emb):
    #if torch.all(emb == 0):
    #    print("All embedding vectors are zeros.")
    #elif torch.any(emb == 0):
    #    print("Some embedding vectors contain zeros.")
    #else:
    #    print("No embedding vectors are entirely zeros.")
    result = []
    for _ in range(100):
        if len(emb) >= 2000:
            idx = torch.randperm(len(emb))[:2000]
        else:
            idx = torch.randperm(len(emb))[:len(emb)]
        emb_cur = emb[idx]
        dists = torch.cdist(emb_cur, emb_cur)
        # Assuming 'dists' is your distance matrix
        #if torch.isinf(dists).any():
        #    print("The distance matrix contains infinite values.")

        #if torch.isnan(dists).any():
        #    print("The distance matrix contains NaN values.")
        delta = delta_hyp(dists)
        diam = dists.max()
        rel_delta = (2 * delta) / diam
        result.append(rel_delta)
    rel_delta_mean = torch.tensor(result).mean().item()
    c = (0.144 / rel_delta_mean) ** 2
    #print(f"δ = {rel_delta_mean:.3f}, c = {c:.3f}")
    return rel_delta_mean, c

def delta_hyp(dismat):
    """
    computes delta hyperbolicity value from distance matrix
    """

    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)

    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)
def batched_delta_hyp(X, n_tries=10, batch_size=1500):
    vals = []
    for i in tqdm(range(n_tries)):
        idx = np.random.choice(len(X), batch_size)
        X_batch = X[idx]
        distmat = distance_matrix(X_batch, X_batch)
        diam = np.max(distmat)
        delta_rel = 2 * delta_hyp(distmat) / diam
        vals.append(delta_rel)
    c = (0.144 / np.mean(vals)) ** 2
    return np.mean(vals), np.std(vals), c

    
