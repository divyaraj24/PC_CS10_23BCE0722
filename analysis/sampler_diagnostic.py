"""Diagnostic: how much does the memory sampler slow down in-process (sequential) runs?

Times the pure sequential K-means (MNIST k=10), k-NN (1000 queries) and preprocessing
pipeline at one size with the sampler in each mode, interleaved, after one warm-up.
Writes results/sampler_diagnostic.csv. Run it on an idle machine:
    python analysis/sampler_diagnostic.py --size 60000 --repeats 3
"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from common import harness
from common.data import ensure_shards, load_mnist
from s01_kmeans.sequential import kmeans_sequential
from s02_knn.sequential import knn_sequential
from s03_preprocess.sequential import preprocess_sequential


def main(n, repeats, out):
    harness.check_env()
    Xtr, ytr, Xte, _ = load_mnist()
    ensure_shards()
    X, y, Q = np.ascontiguousarray(Xtr[:n]), np.ascontiguousarray(ytr[:n]), np.ascontiguousarray(Xte[:1000])
    cases = {"S01 K-means MNIST k=10": lambda: kmeans_sequential(X, 10),
             "S02 k-NN 1000 queries": lambda: knn_sequential(X, y, Q),
             "S03 preprocessing": lambda: preprocess_sequential(n)}
    rows = []
    for case, fn in cases.items():
        fn()  # warm-up
        for rep in range(repeats):
            for mode in ("scan", "track", "off"):  # interleaved to spread drift evenly
                with harness.MemSampler(mode=mode):
                    t = fn()["exec"]
                rows.append({"Case": case, "Problem_Size": n, "Mode": mode, "Rep": rep + 1, "Time": f"{t:.6f}"})
                print(f"{case:28s} {mode:5s} rep{rep + 1} {t:.3f}s", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=60000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(harness.RESULTS / "sampler_diagnostic.csv"))
    a = ap.parse_args()
    main(a.size, a.repeats, a.out)
