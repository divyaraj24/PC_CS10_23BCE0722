"""Pipeline-parallel preprocessing.

loader -> normaliser -> standardiser -> {W HOG replicas} -> main
Each arrow is a multiprocessing.Queue carrying batches of 1000 images. Stages
1-3 are one process each; the HOG stage (the bottleneck) is replicated W times,
so W is the configured worker count (total processes = W + 3).

Each process serialises/deserialises batches explicitly (pickle protocol 5) so
that time can be attributed to Communication; time blocked in Queue.get() is
waiting for upstream (pipeline fill / starvation) -> Synchronization.
"""
import pickle
import time
from multiprocessing import get_context

import numpy as np

from common.data import SHARD_SIZE
from common.harness import track_children
from s03_preprocess import stages as S

QMAX = 8


class _Stats:
    def __init__(self, stage):
        self.d = {"stage": stage, "busy": 0.0, "ser": 0.0, "wait": 0.0}

    def get(self, q):
        t = time.perf_counter(); m = q.get(); self.d["wait"] += time.perf_counter() - t
        if m is None:
            return None
        t = time.perf_counter(); obj = pickle.loads(m); self.d["ser"] += time.perf_counter() - t
        return obj

    def put(self, q, obj):
        t = time.perf_counter(); m = pickle.dumps(obj, protocol=5); self.d["ser"] += time.perf_counter() - t
        q.put(m)


def _loader(n_shards, q_out, q_stats):
    s = _Stats("load")
    for i in range(n_shards):
        t = time.perf_counter(); raw = S.load([i]); s.d["busy"] += time.perf_counter() - t
        s.put(q_out, (i, raw))
    q_out.put(None)
    q_stats.put(s.d)


def _normaliser(q_in, q_out, q_stats):
    s = _Stats("normalise")
    while (m := s.get(q_in)) is not None:
        i, raw = m
        t = time.perf_counter(); X = S.normalise(raw); s.d["busy"] += time.perf_counter() - t
        s.put(q_out, (i, X))
    q_out.put(None)
    q_stats.put(s.d)


def _standardiser(q_in, q_out, n_hog, q_stats):
    s = _Stats("standardise")
    while (m := s.get(q_in)) is not None:
        i, X = m
        t = time.perf_counter(); S.standardise(X); s.d["busy"] += time.perf_counter() - t
        s.put(q_out, (i, X))
    for _ in range(n_hog):
        q_out.put(None)
    q_stats.put(s.d)


def _hog_worker(q_in, q_out, q_stats):
    s = _Stats("hog")
    while (m := s.get(q_in)) is not None:
        i, X = m
        t = time.perf_counter(); F = S.hog_features(X); s.d["busy"] += time.perf_counter() - t
        s.put(q_out, (i, F))
    q_out.put(None)
    q_stats.put(s.d)


def preprocess_pipeline(n, workers):
    ctx = get_context("spawn")
    t0 = time.perf_counter()
    n_shards = n // SHARD_SIZE
    q1, q2, q3 = ctx.Queue(QMAX), ctx.Queue(QMAX), ctx.Queue(QMAX)
    q_out, q_stats = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=_loader, args=(n_shards, q1, q_stats)),
             ctx.Process(target=_normaliser, args=(q1, q2, q_stats)),
             ctx.Process(target=_standardiser, args=(q2, q3, workers, q_stats))]
    procs += [ctx.Process(target=_hog_worker, args=(q3, q_out, q_stats)) for _ in range(workers)]
    ts = time.perf_counter()
    for p in procs:
        p.start()
    sched = time.perf_counter() - ts
    track_children([p.pid for p in procs])

    F = np.empty((n, S.N_FEAT), np.float64)
    main_ser, done = 0.0, 0
    while done < workers:
        m = q_out.get()
        if m is None:
            done += 1
            continue
        t = time.perf_counter()
        i, Fb = pickle.loads(m)
        F[i * SHARD_SIZE:(i + 1) * SHARD_SIZE] = Fb
        main_ser += time.perf_counter() - t
    stats = [q_stats.get() for _ in procs]
    ts = time.perf_counter()
    for p in procs:
        p.join()
    sched += time.perf_counter() - ts
    exec_t = time.perf_counter() - t0

    by_stage = {name: [d for d in stats if d["stage"] == name] for name in S.STAGES}
    busy = {name: sum(d["busy"] for d in ds) for name, ds in by_stage.items()}
    comm = main_ser + sum(max(d["ser"] for d in ds) for ds in by_stage.values())
    sync = max(d["wait"] for d in by_stage["hog"])
    return {"exec": exec_t, "comm": comm, "sync": sync, "sched": sched,
            "result": F, "stages": busy, "stage_kind": "busy_cpu_sum"}
