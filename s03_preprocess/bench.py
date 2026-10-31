"""S03 benchmark driver. Standalone: python -m s03_preprocess.bench --sizes 1000 --workers 1 2"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse

import numpy as np

from common import harness
from common.data import ensure_shards
from s03_preprocess.pipeline import preprocess_pipeline
from s03_preprocess.sequential import preprocess_sequential, preprocess_sequential_chunked
from s03_preprocess.stage_parallel import preprocess_stage_parallel

CHUNKS = 8  # sequential-chunked uses the w=8 decomposition
STAGES_CSV = harness.CsvSink(harness.RESULTS / "s03_stages.csv",
                             ["Run_ID", "Problem_Size", "Parallel_Model", "Resource_Count",
                              "Stage", "Stage_Time", "Time_Kind", "Execution_Time", "Images_per_sec"])


def run(sizes, workers):
    harness.check_env()
    ensure_shards()
    for n in sizes:
        print(f"[S03] images={n}", flush=True)

        def extra(model, w):
            def _w(run_id, out, rep):
                STAGES_CSV.write([{"Run_ID": run_id, "Problem_Size": n, "Parallel_Model": model,
                                   "Resource_Count": w, "Stage": s, "Stage_Time": f"{t:.6f}",
                                   "Time_Kind": out["stage_kind"],
                                   "Execution_Time": f"{out['exec']:.6f}",
                                   "Images_per_sec": f"{n / out['exec']:.2f}"}
                                  for s, t in out["stages"].items()])
            return _w

        ctx = dict(Workload="S03_Preprocess", Dataset="MNIST", Problem_Size=n,
                   Algorithm="load+norm+std+HOG", Parallel_Model="sequential",
                   Resource_Count=1, Threads=1)
        ref = {}
        harness.benchmark(ctx, lambda: preprocess_sequential(n),
                          lambda res, rid: ref.setdefault("F", res), on_timed=extra("sequential", 1))
        Fref = ref["F"]

        def make_verify(cctx):
            def verify(res, run_id):
                ok = res.shape == Fref.shape and np.allclose(res, Fref)
                harness.check(ok, cctx, run_id, "features_allclose",
                              f"maxabs={np.max(np.abs(res - Fref)) if res.shape == Fref.shape else 'shape'}")
            return verify

        cctx = {**ctx, "Parallel_Model": "sequential-chunked"}
        harness.benchmark(cctx, lambda: preprocess_sequential_chunked(n, CHUNKS),
                          make_verify(harness.ctx_keys(cctx)), on_timed=extra("sequential-chunked", 1))
        for model, fn in (("stage-parallel", preprocess_stage_parallel), ("pipeline", preprocess_pipeline)):
            for w in workers:
                pctx = {**ctx, "Parallel_Model": model, "Resource_Count": w}
                harness.benchmark(pctx, lambda: fn(n, w), make_verify(harness.ctx_keys(pctx)),
                                  on_timed=extra(model, w))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 30000, 60000])
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()
    run(a.sizes, a.workers)
