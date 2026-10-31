"""Sequential Lloyd K-means (no Pool). Shared kernels used by the parallel version.

Distances are always computed in fixed, globally aligned blocks of BLOCK rows so
the sequential and parallel versions run byte-identical GEMMs per block.
Partial sums are accumulated in float64.
"""
import time

import numpy as np

BLOCK = 250


def init_centroids(X, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].astype(np.float64)


def tol_scale(X, tol):
    return tol * float(np.mean(np.var(X, axis=0, dtype=np.float64)))


def assign_blocks(X, labels, b0, b1, C32, cnorm):
    """Assign rows of blocks [b0, b1); write labels; return (sums f64, counts, inertia)."""
    k, d = C32.shape
    sums = np.zeros((k, d), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    inertia = 0.0
    n = X.shape[0]
    for b in range(b0, b1):
        s, e = b * BLOCK, min((b + 1) * BLOCK, n)
        Xb = X[s:e]
        dist = cnorm[None, :] - 2.0 * (Xb @ C32.T)          # ||c||^2 - 2 x.c
        lab = np.argmin(dist, axis=1)
        labels[s:e] = lab
        xnorm = np.einsum("ij,ij->i", Xb, Xb, dtype=np.float64)
        inertia += float(np.sum(xnorm + dist[np.arange(e - s), lab]))
        onehot = np.zeros((e - s, k), dtype=np.float32)
        onehot[np.arange(e - s), lab] = 1.0
        sums += (onehot.T @ Xb).astype(np.float64)
        counts += np.bincount(lab, minlength=k)
    return sums, counts, inertia


def update(C, sums, counts):
    Cn = C.copy()
    nz = counts > 0
    Cn[nz] = sums[nz] / counts[nz, None]
    return Cn


def kmeans_sequential(X, k, max_iter=50, tol=1e-4, seed=0):
    t0 = time.perf_counter()
    n = X.shape[0]
    nblocks = (n + BLOCK - 1) // BLOCK
    C = init_centroids(X, k, seed)
    thr = tol_scale(X, tol)
    labels = np.empty(n, dtype=np.int32)
    it, inertia, converged = 0, 0.0, False
    for it in range(1, max_iter + 1):
        C32 = C.astype(np.float32)
        cnorm = np.einsum("ij,ij->i", C32, C32)
        sums, counts, inertia = assign_blocks(X, labels, 0, nblocks, C32, cnorm)
        Cn = update(C, sums, counts)
        shift = float(np.sum((Cn - C) ** 2))
        C = Cn
        if shift <= thr:
            converged = True
            break
    t1 = time.perf_counter()
    return {"exec": t1 - t0, "comm": 0.0, "sync": 0.0, "sched": 0.0,
            "result": {"labels": labels, "centroids": C, "iterations": it,
                       "inertia": inertia, "converged": converged}}


def kmeans_sequential_chunked(X, k, max_iter=50, tol=1e-4, seed=0, n_chunks=8):
    """Diagnostic: the parallel decomposition (n_chunks contiguous block ranges, one
    partial result per range, reduced in range order) executed in-process, no Pool."""
    from common.harness import split_ranges
    t0 = time.perf_counter()
    n = X.shape[0]
    nblocks = (n + BLOCK - 1) // BLOCK
    ranges = split_ranges(nblocks, n_chunks)
    C = init_centroids(X, k, seed)
    thr = tol_scale(X, tol)
    labels = np.empty(n, dtype=np.int32)
    it, inertia, converged = 0, 0.0, False
    for it in range(1, max_iter + 1):
        C32 = C.astype(np.float32)
        cnorm = np.einsum("ij,ij->i", C32, C32)
        parts = [assign_blocks(X, labels, b0, b1, C32, cnorm) for b0, b1 in ranges]
        sums, counts, inertia = parts[0][0].copy(), parts[0][1].copy(), parts[0][2]
        for p in parts[1:]:
            sums += p[0]
            counts += p[1]
            inertia += p[2]
        Cn = update(C, sums, counts)
        shift = float(np.sum((Cn - C) ** 2))
        C = Cn
        if shift <= thr:
            converged = True
            break
    t1 = time.perf_counter()
    return {"exec": t1 - t0, "comm": 0.0, "sync": 0.0, "sched": 0.0,
            "result": {"labels": labels, "centroids": C, "iterations": it,
                       "inertia": inertia, "converged": converged}}
