#!/usr/bin/env python3
# how much of the original weights each 50% pruning rule keeps, on the same random weights.
# a stand-in for "how much accuracy might pruning cost": it is NOT a trained-model accuracy test.
#   2:4         every group of 4 keeps its 2 largest
#   paired 4:8  every group of 8 = 4 neighbouring pairs, keeps the 2 pairs with the most combined size (NVIDIA, FP4 only)
#   4:8         every group of 8 keeps its 4 largest (this project)
#   no pattern  the best possible 50%: the largest half of each whole column (reference, not hardware friendly)
# metrics: share of the weights' total energy (sum of squares) kept, and how far the answer x.W moves.
#
# run: ~/chipyard/.conda-env/bin/python verification/pattern_fidelity.py
import numpy as np

rng = np.random.default_rng(7)
K, J, X = 4096, 256, 64          # weights K x J (pruned down K, per column like the hardware), X input rows


def keep_mask(w, rule):
    k, j = w.shape
    a = w * w
    m = np.zeros_like(w, dtype=bool)
    if rule == "2:4":
        g = a.reshape(k // 4, 4, j)
        top = np.argsort(-g, axis=1, kind="stable")[:, :2, :]
        mm = np.zeros_like(g, dtype=bool); np.put_along_axis(mm, top, True, axis=1); m = mm.reshape(k, j)
    elif rule == "4:8":
        g = a.reshape(k // 8, 8, j)
        top = np.argsort(-g, axis=1, kind="stable")[:, :4, :]
        mm = np.zeros_like(g, dtype=bool); np.put_along_axis(mm, top, True, axis=1); m = mm.reshape(k, j)
    elif rule == "paired 4:8":
        g = a.reshape(k // 8, 4, 2, j).sum(axis=2)              # energy of each neighbouring pair
        top = np.argsort(-g, axis=1, kind="stable")[:, :2, :]
        pm = np.zeros_like(g, dtype=bool); np.put_along_axis(pm, top, True, axis=1)
        m = np.repeat(pm, 2, axis=1).reshape(k, j)
    elif rule == "no pattern":
        top = np.argsort(-a, axis=0, kind="stable")[: k // 2, :]
        np.put_along_axis(m, top, True, axis=0)
    assert m.sum() == w.size // 2
    return m


print("share of weights kept is 50% for every rule; the question is WHICH half")
print(f"{'weights':<10} {'rule':<12} {'energy kept':>12} {'answer error':>13}")
results = {}
for dist in ["bell curve", "heavy tail"]:
    w = rng.standard_normal((K, J)) if dist == "bell curve" else rng.laplace(size=(K, J))
    x = rng.standard_normal((X, K))
    ref = x @ w
    for rule in ["paired 4:8", "2:4", "4:8", "no pattern"]:
        wp = np.where(keep_mask(w, rule), w, 0)
        kept = (wp ** 2).sum() / (w ** 2).sum()
        err = np.linalg.norm(x @ wp - ref) / np.linalg.norm(ref)
        results[(dist, rule)] = (kept, err)
        print(f"{dist:<10} {rule:<12} {kept * 100:11.1f}% {err * 100:12.1f}%")
# the allowed-patterns count per group of 8 weights
from math import comb
print(f"allowed patterns per 8 weights: paired 4:8 = {comb(4, 2)}, 2:4 = {comb(4, 2) ** 2}, 4:8 = {comb(8, 4)}")
