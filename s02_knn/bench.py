"""S02 benchmark driver.
Standalone: python -m s02_knn.bench --sizes 1000 --queries 1000 --workers 1 2

Problem_Size = training-set size; the query count is in Dataset ("MNIST-q5000");
the strategy is in Algorithm ("kNN-brute-k5/query-partitioned").
"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse

import numpy as np

from common import harness
from common.data import load_mnist
from s02_knn.parallel import STRATEGIES, knn_parallel
from s02_knn.sequential import K, knn_sequential, knn_sequential_chunked

QUERIES = [1000, 5000, 10000]
CHUNKS = 8  # sequential-chunked uses the w=8 query decomposition
BASE_ALGO = f"kNN-brute-k{K}"
EXTRA = harness.CsvSink(harness.RESULTS / "s02_knn.csv",
                        ["Run_ID", "Problem_Size", "N_Queries", "Strategy", "Parallel_Model",
                         "Resource_Count", "Search_Time", "Queries_per_sec", "Accuracy"])


def run(sizes, workers, queries=None):
    harness.check_env()
    Xtr, ytr, Xte, yte = load_mnist()
    for n in sizes:
        X, y = np.ascontiguousarray(Xtr[:n]), np.ascontiguousarray(ytr[:n])
        for nq in (queries or QUERIES):
            Q, yq = np.ascontiguousarray(Xte[:nq]), yte[:nq]
            print(f"[S02] train={n} queries={nq} k={K}", flush=True)

            def extra(model, w, strategy):
                def _w(run_id, out, rep):
                    EXTRA.write([{"Run_ID": run_id, "Problem_Size": n, "N_Queries": nq,
                                  "Strategy": strategy, "Parallel_Model": model, "Resource_Count": w,
                                  "Search_Time": f"{out['exec']:.6f}",
                                  "Queries_per_sec": f"{nq / out['exec']:.3f}",
                                  "Accuracy": f"{float(np.mean(out['result'] == yq)):.4f}"}])
                return _w

            ctx = dict(Workload="S02_kNN", Dataset=f"MNIST-q{nq}", Problem_Size=n,
                       Algorithm=BASE_ALGO, Parallel_Model="sequential", Resource_Count=1, Threads=1)
            ref = {}
            harness.benchmark(ctx, lambda: knn_sequential(X, y, Q),
                              lambda res, rid: ref.setdefault("p", res),
                              on_timed=extra("sequential", 1, "none"))
            P = ref["p"]

            def make_verify(cctx):
                def verify(res, run_id):
                    nd = int(np.sum(res != P))
                    harness.check(nd == 0, cctx, run_id, "predictions_identical", f"mismatches={nd}")
                    acc_p, acc_s = float(np.mean(res == yq)), float(np.mean(P == yq))
                    harness.check(acc_p == acc_s, cctx, run_id, "accuracy_equal",
                                  f"par={acc_p:.4f} seq={acc_s:.4f}")
                return verify

            cctx = {**ctx, "Parallel_Model": "sequential-chunked"}
            harness.benchmark(cctx, lambda: knn_sequential_chunked(X, y, Q, CHUNKS),
                              make_verify(harness.ctx_keys(cctx)),
                              on_timed=extra("sequential-chunked", 1, "query-partitioned"))
            for strategy in STRATEGIES:
                for w in workers:
                    pctx = {**ctx, "Algorithm": f"{BASE_ALGO}/{strategy}",
                            "Parallel_Model": "multiprocessing", "Resource_Count": w}
                    harness.benchmark(pctx, lambda: knn_parallel(X, y, Q, w, strategy),
                                      make_verify(harness.ctx_keys(pctx)),
                                      on_timed=extra("multiprocessing", w, strategy))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 30000, 60000])
    ap.add_argument("--queries", type=int, nargs="+", default=QUERIES)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()
    run(a.sizes, a.workers, a.queries)
