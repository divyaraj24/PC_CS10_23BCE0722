"""S01 benchmark driver. Standalone: python -m s01_kmeans.bench --sizes 1000 --workers 1 2"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse

import numpy as np

from common import harness
from common.data import load_mnist, make_blobs_data
from s01_kmeans.parallel import kmeans_parallel
from s01_kmeans.sequential import kmeans_sequential, kmeans_sequential_chunked

EXTRA = harness.CsvSink(harness.RESULTS / "s01_kmeans.csv",
                        ["Run_ID", "Dataset", "Problem_Size", "k", "Parallel_Model",
                         "Resource_Count", "Iterations", "Converged", "Inertia"])
MAX_ITER, TOL = 50, 1e-4
CHUNKS = 8  # sequential-chunked uses the w=8 decomposition


def configs(sizes):
    """(dataset, size, k) triples: k=10 everywhere, k=5/20 at the middle size."""
    out = []
    for ds in ("MNIST", "Blobs64d"):
        for n in sizes:
            out.append((ds, n, 10))
    mid = sizes[len(sizes) // 2]
    out += [("MNIST", mid, 5), ("MNIST", mid, 20)]
    return out


def run(sizes, workers):
    harness.check_env()
    Xtr = load_mnist()[0]
    for ds, n, k in configs(sizes):
        X = np.ascontiguousarray(Xtr[:n]) if ds == "MNIST" else make_blobs_data(n)
        algo = f"KMeans-Lloyd-k{k}"
        print(f"[S01] {ds} n={n} k={k}", flush=True)

        def extra(model, w):
            def _w(run_id, out, rep):
                r = out["result"]
                EXTRA.write([{"Run_ID": run_id, "Dataset": ds, "Problem_Size": n, "k": k,
                              "Parallel_Model": model, "Resource_Count": w,
                              "Iterations": r["iterations"], "Converged": r["converged"],
                              "Inertia": f"{r['inertia']:.6f}"}])
            return _w

        ctx = dict(Workload="S01_KMeans", Dataset=ds, Problem_Size=n, Algorithm=algo,
                   Parallel_Model="sequential", Resource_Count=1, Threads=1)
        ref = {}

        def keep(res, run_id):
            ref.setdefault("r", res)

        harness.benchmark(ctx, lambda: kmeans_sequential(X, k, MAX_ITER, TOL), keep,
                          on_timed=extra("sequential", 1))
        R = ref["r"]

        def make_verify(cctx):
            def verify(res, run_id):
                harness.check(res["iterations"] == R["iterations"], cctx, run_id, "iterations_equal",
                              f"par={res['iterations']} seq={R['iterations']}")
                nd = int(np.sum(res["labels"] != R["labels"]))
                harness.check(nd == 0, cctx, run_id, "labels_identical", f"mismatches={nd}")
                harness.check(np.allclose(res["centroids"], R["centroids"], rtol=1e-6, atol=1e-8),
                              cctx, run_id, "centroids_allclose",
                              f"maxabs={np.max(np.abs(res['centroids'] - R['centroids'])):.3e}")
            return verify

        # diagnostic baseline: the 8-worker chunk decomposition run in-process
        cctx_ = {**ctx, "Parallel_Model": "sequential-chunked"}
        harness.benchmark(cctx_, lambda: kmeans_sequential_chunked(X, k, MAX_ITER, TOL, n_chunks=CHUNKS),
                          make_verify(harness.ctx_keys(cctx_)), on_timed=extra("sequential-chunked", 1))

        for w in workers:
            pctx = {**ctx, "Parallel_Model": "multiprocessing", "Resource_Count": w}
            harness.benchmark(pctx, lambda: kmeans_parallel(X, k, w, MAX_ITER, TOL),
                              make_verify(harness.ctx_keys(pctx)), on_timed=extra("multiprocessing", w))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 30000, 60000])
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()
    run(a.sizes, a.workers)
