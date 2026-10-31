"""Parallel k-NN over a Pool; train set, labels, queries and norms in shared memory.

query-partitioned: query blocks split across workers; each worker returns predictions.
train-partitioned: train chunks split across workers; each worker returns its local
                   top-k (distances + global indices) for every query; the parent
                   merges to the global top-k and votes.
"""
import time
from multiprocessing import get_context

import numpy as np

from common.harness import Overheads, SharedArray, pool_init, split_ranges, timed_map, track_pool, warr
from s02_knn.sequential import K, QBLOCK, merge_topk, n_chunks, predict_blocks, topk_train_part, vote

STRATEGIES = ("query-partitioned", "train-partitioned")


def _query_task(args):
    b0, b1 = args
    return predict_blocks(warr("Q"), warr("X"), warr("xnorm"), warr("y"), b0, b1)


def _train_task(args):
    c0, c1 = args
    return topk_train_part(warr("Q"), warr("X"), warr("xnorm"), c0, c1)


def knn_parallel(X, y, Q, workers, strategy="query-partitioned"):
    ov = Overheads()
    t0 = time.perf_counter()
    xnorm = np.einsum("ij,ij->i", X, X)

    tc = time.perf_counter()
    shared = {"X": SharedArray.from_array(X), "y": SharedArray.from_array(y),
              "Q": SharedArray.from_array(Q), "xnorm": SharedArray.from_array(xnorm)}
    ov.comm += time.perf_counter() - tc

    ts = time.perf_counter()
    pool = get_context("spawn").Pool(workers, initializer=pool_init,
                                     initargs=({k: v.spec for k, v in shared.items()},))
    ov.sched += time.perf_counter() - ts
    track_pool(pool)
    try:
        if strategy == "query-partitioned":
            nblocks = (Q.shape[0] + QBLOCK - 1) // QBLOCK
            preds = np.concatenate(timed_map(pool, _query_task, split_ranges(nblocks, workers), ov))
        elif strategy == "train-partitioned":
            parts = timed_map(pool, _train_task, split_ranges(n_chunks(X.shape[0]), workers), ov)
            _, nn = merge_topk([p[0] for p in parts], [p[1] for p in parts], K)  # parent-side reduction
            preds = vote(nn, y)
        else:
            raise ValueError(strategy)
    finally:
        ts = time.perf_counter()
        pool.close()
        pool.join()
        ov.sched += time.perf_counter() - ts
        for v in shared.values():
            v.release()
    t1 = time.perf_counter()
    return {"exec": t1 - t0, "comm": ov.comm, "sync": ov.sync, "sched": ov.sched, "result": preds}
