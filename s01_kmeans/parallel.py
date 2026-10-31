"""Parallel K-means: assignment step data-parallel over a persistent Pool.

X lives in shared memory (copied once per run). Each iteration the current
centroids are broadcast as task arguments; each worker assigns its contiguous
range of blocks, writes labels into shared memory and returns partial sums +
counts + inertia. The parent reduces the partials and updates centroids.
"""
import time
from multiprocessing import get_context

import numpy as np

from common.harness import Overheads, track_pool, SharedArray, pool_init, split_ranges, timed_map, warr
from s01_kmeans.sequential import BLOCK, assign_blocks, init_centroids, tol_scale, update


def _assign_task(args):
    b0, b1, C32, cnorm = args
    return assign_blocks(warr("X"), warr("labels"), b0, b1, C32, cnorm)


def kmeans_parallel(X, k, workers, max_iter=50, tol=1e-4, seed=0):
    ov = Overheads()
    t0 = time.perf_counter()
    n = X.shape[0]
    nblocks = (n + BLOCK - 1) // BLOCK
    ranges = split_ranges(nblocks, workers)

    tc = time.perf_counter()
    sX = SharedArray.from_array(X)
    sL = SharedArray((n,), np.int32)
    ov.comm += time.perf_counter() - tc

    ts = time.perf_counter()
    pool = get_context("spawn").Pool(workers, initializer=pool_init,
                                     initargs=({"X": sX.spec, "labels": sL.spec},))
    ov.sched += time.perf_counter() - ts
    track_pool(pool)
    try:
        C = init_centroids(X, k, seed)
        thr = tol_scale(X, tol)
        it, inertia, converged = 0, 0.0, False
        for it in range(1, max_iter + 1):
            C32 = C.astype(np.float32)
            cnorm = np.einsum("ij,ij->i", C32, C32)
            parts = timed_map(pool, _assign_task, [(b0, b1, C32, cnorm) for b0, b1 in ranges], ov)
            sums = parts[0][0].copy()
            counts = parts[0][1].copy()
            inertia = parts[0][2]
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
        tc = time.perf_counter()
        labels = sL.arr.copy()
        ov.comm += time.perf_counter() - tc
    finally:
        ts = time.perf_counter()
        pool.close()
        pool.join()
        ov.sched += time.perf_counter() - ts
        sX.release()
        sL.release()
    t1 = time.perf_counter()
    return {"exec": t1 - t0, "comm": ov.comm, "sync": ov.sync, "sched": ov.sched,
            "result": {"labels": labels, "centroids": C, "iterations": it,
                       "inertia": inertia, "converged": converged}}
