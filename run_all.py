"""Run every experiment and write results/raw.csv (+ side CSVs, env.txt).

    python run_all.py                       # full sweep
    python run_all.py --smoke               # 1k samples, workers 1 2 (writes to results_smoke/)
    python run_all.py --only s01 s02        # subset
"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import sys
import time
from pathlib import Path

from common import harness

SIZES = [10000, 30000, 60000]
WORKERS = [1, 2, 4, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+")
    ap.add_argument("--workers", type=int, nargs="+")
    ap.add_argument("--queries", type=int, nargs="+", help="S02 query counts (default 1000 5000 10000)")
    ap.add_argument("--only", nargs="+", default=["s01", "s02", "s03"])
    ap.add_argument("--mem-sampler", choices=["track", "scan", "off"],
                    help="memory sampler mode (default: track; env CS10_MEM_SAMPLER)")
    ap.add_argument("--keep", action="store_true", help="append instead of clearing old CSVs")
    a = ap.parse_args()
    if a.mem_sampler:
        harness.MEM_MODE = a.mem_sampler
    sizes = a.sizes or ([1000] if a.smoke else SIZES)
    workers = a.workers or ([1, 2] if a.smoke else WORKERS)
    if a.smoke:
        out = harness.ROOT / "results_smoke"
        for sink in (harness.RAW, harness.CORRECTNESS):
            sink.path = out / sink.path.name
        harness.RESULTS = out

    # import after RESULTS may have been redirected (side-CSV sinks bind at import)
    from s01_kmeans import bench as b1
    from s02_knn import bench as b2
    from s03_preprocess import bench as b3
    mods = {"s01": b1, "s02": b2, "s03": b3}
    if a.smoke:
        for m in mods.values():
            for name in ("EXTRA", "STAGES_CSV"):
                if hasattr(m, name):
                    sink = getattr(m, name)
                    sink.path = harness.RESULTS / sink.path.name
    if not a.keep:
        harness.RAW.reset(); harness.CORRECTNESS.reset()
        for m in mods.values():
            for name in ("EXTRA", "STAGES_CSV"):
                if hasattr(m, name):
                    getattr(m, name).reset()

    harness.RESULTS.mkdir(parents=True, exist_ok=True)
    harness.write_env(harness.RESULTS / "env.txt")
    print(f"sizes={sizes} workers={workers} -> {harness.RAW.path}", flush=True)
    timings = {}
    for key in a.only:
        t = time.perf_counter()
        try:
            if key == "s02":
                mods[key].run(sizes, workers, a.queries)
            else:
                mods[key].run(sizes, workers)
        except harness.CorrectnessError as e:
            print(f"\n!!! {e}\n!!! Stopping. See {harness.CORRECTNESS.path}", file=sys.stderr)
            sys.exit(2)
        timings[key] = time.perf_counter() - t
        print(f"== {key} done in {timings[key]:.1f}s", flush=True)
    print("phase times (s):", {k: round(v, 1) for k, v in timings.items()},
          "total:", round(sum(timings.values()), 1))


if __name__ == "__main__":
    main()
