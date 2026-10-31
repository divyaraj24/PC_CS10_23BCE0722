"""Sequential preprocessing (no Pool), each stage timed separately."""
import time

from common.data import SHARD_SIZE
from s03_preprocess import stages as S


def preprocess_sequential(n):
    st = {}
    t0 = time.perf_counter()
    t = time.perf_counter(); raw = S.load(range(n // SHARD_SIZE)); st["load"] = time.perf_counter() - t
    t = time.perf_counter(); X = S.normalise(raw); st["normalise"] = time.perf_counter() - t
    t = time.perf_counter(); S.standardise(X); st["standardise"] = time.perf_counter() - t
    t = time.perf_counter(); F = S.hog_features(X); st["hog"] = time.perf_counter() - t
    return {"exec": time.perf_counter() - t0, "comm": 0.0, "sync": 0.0, "sched": 0.0,
            "result": F, "stages": st, "stage_kind": "wall"}


def preprocess_sequential_chunked(n, n_chunks=8):
    """Diagnostic: the 8-worker stage-parallel decomposition (same shard / row ranges per
    stage, writing into preallocated output arrays) executed in-process, no Pool."""
    import numpy as np
    from common.harness import split_ranges
    st = {}
    t0 = time.perf_counter()
    raw = np.empty((n, 784), np.uint8)
    X = np.empty((n, 784), np.float32)
    F = np.empty((n, S.N_FEAT), np.float64)
    rows = split_ranges(n, n_chunks)
    t = time.perf_counter()
    for s, e in split_ranges(n // SHARD_SIZE, n_chunks):
        raw[s * SHARD_SIZE:e * SHARD_SIZE] = S.load(range(s, e))
    st["load"] = time.perf_counter() - t
    t = time.perf_counter()
    for s, e in rows:
        S.normalise(raw[s:e], out=X[s:e])
    st["normalise"] = time.perf_counter() - t
    t = time.perf_counter()
    for s, e in rows:
        S.standardise(X[s:e])
    st["standardise"] = time.perf_counter() - t
    t = time.perf_counter()
    for s, e in rows:
        S.hog_features(X[s:e], out=F[s:e])
    st["hog"] = time.perf_counter() - t
    return {"exec": time.perf_counter() - t0, "comm": 0.0, "sync": 0.0, "sched": 0.0,
            "result": F, "stages": st, "stage_kind": "wall"}
