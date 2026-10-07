#!/usr/bin/env python3
# independent check of the n:m sparse math, written from the spec in numpy.
# shares no code with the C helpers in gemmini_nm.h or the spike model.
#
# what the sparse array computes for one squashed DIM x DIM weight tile (from the rtl):
#   pe row r holds weight wc[r, j] and index idx[r, j]; it belongs to group g = r // n
#   and multiplies the activation at column g*m + idx[r, j] of a 2*DIM wide activation row
#   out[i, j] = sum_r a[i, (r // n) * m + idx[r, j]] * wc[r, j]
# that has to equal the ordinary dense product a @ w_pruned.
#
# run: ~/chipyard/.conda-env/bin/python verification/independent_math_check.py
import numpy as np

DIM = 16
rng = np.random.default_rng(20261006)


def prune(w, n, m):
    """keep the n largest-magnitude values in every group of m rows, per column"""
    w = w.copy()
    k, j = w.shape
    for g in range(0, k, m):
        blk = w[g:g + m]
        # stable sort so ties go to the lower row, like the C code
        order = np.argsort(-np.abs(blk), axis=0, kind="stable")
        drop = order[n:]
        for c in range(j):
            blk[drop[:, c], c] = 0
    return w


def compress(w, n, m):
    """-> (wc, idx): n kept values per group of m, plus the row they came from inside the group"""
    k, j = w.shape
    wc = np.zeros((k // m * n, j), dtype=w.dtype)
    idx = np.zeros_like(wc, dtype=np.int64)
    for g in range(k // m):
        for c in range(j):
            nz = [r for r in range(m) if w[g * m + r, c] != 0][:n]
            pad = [r for r in range(m) if r not in nz][: n - len(nz)]
            rows = sorted(nz + pad)
            for s, r in enumerate(rows):
                wc[g * n + s, c] = w[g * m + r, c]
                idx[g * n + s, c] = r
    return wc, idx


def sparse_array_tile(a2, wc, idx, n, m, how="correct"):
    """what the hardware does for one tile; `how` injects the failure modes nm_sparse_debug looks for"""
    out = np.zeros((a2.shape[0], wc.shape[1]), dtype=np.int64)
    for r in range(DIM):
        g = r // n
        for c in range(DIM):
            i = idx[r, c]
            if how == "index ignored": i = 0
            if how == "index rows flipped": i = idx[DIM - 1 - r, c]
            if how == "one index wrong" and (r, c) == (3, 5): i = (i + 1) % m
            col = g * m + i
            if how == "high half missing" and col >= DIM: continue
            out[:, c] += a2[:, col].astype(np.int64) * wc[r, c]
    return out


def check_pattern(n, m, trials=500):
    assert m == 2 * n
    worst = 0
    for t in range(trials):
        a2 = rng.integers(-128, 128, size=(DIM, 2 * DIM), dtype=np.int64)   # int8 activations
        w = rng.integers(-128, 128, size=(2 * DIM, DIM), dtype=np.int64)    # int8 weights
        if t % 5 == 0:                       # some groups with fewer than n nonzeros
            w[rng.random(w.shape) < 0.6] = 0
        wp = prune(w, n, m)
        # every group really has at most n nonzeros
        assert all((wp[g:g + m] != 0).sum(axis=0).max() <= n for g in range(0, 2 * DIM, m))
        wc, idx = compress(wp, n, m)
        dense = a2 @ wp
        sparse = sparse_array_tile(a2, wc, idx, n, m)
        assert np.array_equal(dense, sparse), f"{n}:{m} trial {t} mismatch"
        worst = max(worst, int(np.abs(dense).max()))
    return worst


def negative_controls(n, m, trials=200):
    """the comparison has to notice each kind of hardware mistake"""
    caught = {}
    for how in ["index ignored", "index rows flipped", "one index wrong", "high half missing"]:
        hits = 0
        for _ in range(trials):
            a2 = rng.integers(-128, 128, size=(DIM, 2 * DIM), dtype=np.int64)
            w = rng.integers(-128, 128, size=(2 * DIM, DIM), dtype=np.int64)
            wp = prune(w, n, m)
            wc, idx = compress(wp, n, m)
            if not np.array_equal(a2 @ wp, sparse_array_tile(a2, wc, idx, n, m, how)):
                hits += 1
        caught[how] = hits
    return caught


if __name__ == "__main__":
    print("independent n:m math check (numpy, no shared code with the C tests)")
    for n, m in [(2, 4), (4, 8)]:
        worst = check_pattern(n, m)
        print(f"  {n}:{m}  500 random 16x32 by 32x16 tiles: sparse array math == dense a @ w_pruned in every case"
              f" (largest |output| {worst}, fits int32)")
        for how, hits in negative_controls(n, m).items():
            print(f"  {n}:{m}  injected fault '{how}': detected in {hits}/200 trials")
    # tile counts behind the 2x array-level claim (nm_sparse_perf is 64x128x32)
    I, K, J = 64, 128, 32
    dense_tiles = (I // DIM) * (K // DIM) * (J // DIM)
    sparse_tiles = (I // DIM) * (K // 2 // DIM) * (J // DIM)
    print(f"  64x128x32 matmul: dense needs {dense_tiles} weight-tile passes, sparse needs {sparse_tiles}")
