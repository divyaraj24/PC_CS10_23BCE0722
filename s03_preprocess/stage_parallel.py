"""Stage-parallel preprocessing: each stage is data-parallel over one Pool, with a
barrier (pool.map) between stages. Arrays live in shared memory."""
import time
from multiprocessing import get_context

import numpy as np

from common.data import SHARD_SIZE
from common.harness import Overheads, track_pool, SharedArray, pool_init, split_ranges, timed_map, warr
from s03_preprocess import stages as S


def _load(r):
    s, e = r
    warr("raw")[s * SHARD_SIZE:e * SHARD_SIZE] = S.load(range(s, e))


def _normalise(r):
    s, e = r
    S.normalise(warr("raw")[s:e], out=warr("X")[s:e])


def _standardise(r):
    s, e = r
    S.standardise(warr("X")[s:e])


def _hog(r):
    s, e = r
    S.hog_features(warr("X")[s:e], out=warr("F")[s:e])


def preprocess_stage_parallel(n, workers):
    ov = Overheads()
    st = {}
    t0 = time.perf_counter()
    tc = time.perf_counter()
    sh = {"raw": SharedArray((n, 784), np.uint8), "X": SharedArray((n, 784), np.float32),
          "F": SharedArray((n, S.N_FEAT), np.float64)}
    ov.comm += time.perf_counter() - tc
    ts = time.perf_counter()
    pool = get_context("spawn").Pool(workers, initializer=pool_init,
                                     initargs=({k: v.spec for k, v in sh.items()},))
    ov.sched += time.perf_counter() - ts
    track_pool(pool)
    try:
        rows = split_ranges(n, workers)
        plan = [("load", _load, split_ranges(n // SHARD_SIZE, workers)),
                ("normalise", _normalise, rows), ("standardise", _standardise, rows),
                ("hog", _hog, rows)]
        for name, fn, ranges in plan:
            t = time.perf_counter()
            timed_map(pool, fn, ranges, ov)
            st[name] = time.perf_counter() - t
        tc = time.perf_counter()
        F = sh["F"].arr.copy()
        ov.comm += time.perf_counter() - tc
    finally:
        ts = time.perf_counter()
        pool.close()
        pool.join()
        ov.sched += time.perf_counter() - ts
        for v in sh.values():
            v.release()
    return {"exec": time.perf_counter() - t0, "comm": ov.comm, "sync": ov.sync,
            "sched": ov.sched, "result": F, "stages": st, "stage_kind": "wall"}
