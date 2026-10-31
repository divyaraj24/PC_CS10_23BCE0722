# CS10 — Parallel Machine Learning (BCSE412L case study)

Divyaraj (23BCE0722), batch CS10. BCSE412L Parallel Computing, Fall Semester 2026-27, VIT. CPU-only benchmarks of three ML workloads with Python
`multiprocessing` on an Apple M1 MacBook Air.

| Sub-problem | Sequential baseline | Parallel variant(s) |
|---|---|---|
| S01 K-means (Lloyd) | `s01_kmeans/sequential.py` | `s01_kmeans/parallel.py`: assignment step data-parallel over a persistent Pool |
| S02 brute-force k-NN | `s02_knn/sequential.py` | `s02_knn/parallel.py`: **query-partitioned** (query blocks split over a Pool) and **train-partitioned** (training set split; each worker returns a local top-k per query; the parent merges to the global top-k and votes) |
| S03 preprocessing (load → normalise → standardise → HOG) | `s03_preprocess/sequential.py` | `stage_parallel.py` (each stage data-parallel, barrier between stages), `pipeline.py` (stages as processes joined by Queues, batches of 1000) |

## Reproduce

```bash
git clone https://github.com/divyaraj24/PC_CS10_23BCE0722.git
cd PC_CS10_23BCE0722
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m common.data          # one-off: download MNIST -> data/mnist.npz, write data/shards/
.venv/bin/python run_all.py --smoke      # optional: 1k samples, workers 1 2 -> results_smoke/ (~30 s)
.venv/bin/python analysis/sampler_diagnostic.py   # optional, ~2 min: memory-sampler cost on sequential runs
.venv/bin/python run_all.py              # full sweep -> results/raw.csv etc. (~25 min on an M1 Air)
.venv/bin/python analysis/analyze.py     # derived.csv, config_stats.csv, summary_table.csv/.md
.venv/bin/python analysis/plots.py       # plots/*.png (300 dpi) + plots/*.svg
.venv/bin/python analysis/findings.py    # results/findings.md
```

`run_all.py` options:

| Option | Effect |
|---|---|
| `--smoke` | 1k samples, workers 1 and 2, written to `results_smoke/` |
| `--only s01 s02 s03` | run a subset of the sub-problems |
| `--sizes N ...` / `--workers W ...` | override problem sizes (default 10000 30000 60000) and worker counts (default 1 2 4 8) |
| `--queries Q ...` | S02 query counts (default 1000 5000 10000) |
| `--mem-sampler track\|scan\|off` | memory sampler mode (default `track`; also env `CS10_MEM_SAMPLER`) |
| `--keep` | append to the existing CSVs instead of clearing them |

The analysis scripts take `--results DIR` (and `plots.py` also takes `--out DIR`), so they can be run
on the smoke output, e.g. `analysis/analyze.py --results results_smoke`. You can also run one
sub-problem directly, e.g. `python -m s02_knn.bench --sizes 10000 --queries 1000 --workers 1 2 4 8`.

Before the final sweep, quit other apps, wait for Spotlight/cloud sync to go quiet (`top -l 2 -o cpu`),
check that low power mode is off (`pmset -g | grep lowpowermode`), and write what you find to
`results/preflight.txt`. `run_all.py` appends that file to `results/env.txt`.

## Experimental design

- **Threads pinned.** Every entry script sets `OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 before importing numpy. Spawned workers
  inherit this, so the sequential baseline really uses one core. NumPy uses Apple Accelerate as its BLAS.
- **Workers** 1, 2, 4 and 8 (spawn start method), plus a pure sequential baseline that creates no Pool.
  The M1 has 4 performance and 4 efficiency cores, so the 8-worker runs use the efficiency cores too.
- **Repeats.** Each configuration gets 1 warm-up run (discarded) and 5 timed runs.
- **Sizes.** 10k, 30k and 60k samples. For S01: MNIST and synthetic blobs (64 features, 10 centres)
  with k=10, plus k=5 and k=20 on MNIST at 30k. For S02: 10k, 30k and 60k training images ×
  1000, 5000 and 10000 queries (the first N MNIST test images), k=5. Problem_Size = training size;
  the query count is in Dataset (`MNIST-q5000`), and the strategy in Algorithm
  (`kNN-brute-k5/train-partitioned`). For S03: the first 10k, 30k or 60k MNIST training images.
- **Shared memory.** Datasets and output arrays are placed in `multiprocessing.shared_memory`
  once per run, and workers attach to them in the Pool initializer. Each task carries only
  index ranges (plus the centroids for K-means).
- **Speedup baseline.** Speedup is always measured against the pure sequential implementation at the
  same problem size (mean of its 5 timed runs). Both k-NN strategies share the same baseline.
  `derived.csv` records this in the `Baseline` column. Efficiency = speedup / Resource_Count.
- **Compute-only speedup** (analysis only) = sequential time / (Execution_Time − Scheduling_Time).
  It removes Pool/process start-up, shutdown and dispatch latency, and is reported alongside the
  end-to-end speedup.
- **`sequential-chunked` (diagnostic, not a baseline).** This is the 8-worker task decomposition (same
  chunk ranges, same task kernel, same reduction order) executed in-process with no Pool. It is used
  to test whether chunking/cache locality explains w=1 runs beating pure sequential. findings.md has
  the verdict table. At w=1 a worker's chunk is the whole problem, so its chunking already matches
  sequential.

### Timing decomposition (per run)

| Column | Meaning |
|---|---|
| Execution_Time | Total wall time of the algorithm, including shared-memory setup, Pool/process start-up and shutdown. Dataset download and disk caching are excluded; S03 loading from `.npz` is part of the workload. |
| Scheduling_Time | Pool creation, `close()`/`join()`, and per-`map` dispatch latency (from submit to the first task starting). For the pipeline: `Process.start()` and `join()`. |
| Communication_Time | Copying data into shared memory and copying results out, plus result return latency per `map` (from the last task ending to the parent receiving results). For the pipeline: explicit pickle serialisation and deserialisation of batches in every stage, plus reassembly in the parent. |
| Synchronization_Time | Barrier idle time per `map`: (last end − first start) − busiest worker's compute time. For the pipeline: the longest time any HOG replica spent blocked waiting for input (pipeline fill or starvation). |
| Memory_Use | Peak RSS in MB of the parent plus its registered worker processes, sampled every 20 ms by a thread in the parent. It includes the parent's in-memory copy of MNIST (about 400 MB). |
| Threads | BLAS threads per process (always 1). |
| Energy_if_available | NA (not measured). |

**Memory sampler modes.** The first sweep used `scan` (`psutil` `children(recursive=True)`). On macOS
that scans every process on the system, which costs about 3 ms per sample, every 20 ms, while holding
the GIL in the parent process. Only in-process (sequential) runs are slowed by it, so the baselines
were biased upward. The final sweep uses `track`: it reads only the PIDs of the Pool workers and
pipeline processes (about 0.1 ms per sample). `analysis/sampler_diagnostic.py` measures sequential
runs under `scan`, `track` and `off`, and findings.md reports the result.

Stamps taken across processes use `time.time()`. Everything else uses `time.perf_counter()`.

### Correctness (asserted on every run, including warm-ups → `results/correctness.csv`)

- **K-means:** iteration count equal to sequential, cluster labels identical, and centroids
  `allclose`. Initial centroids come from the same seeded sample (seed 0). Distances are
  computed in fixed, globally aligned 250-row blocks, so sequential and parallel runs execute
  identical GEMMs. Partial sums are accumulated in float64.
- **k-NN:** predictions identical to sequential, and accuracy equal, for both strategies. Query
  blocks (125 rows) and train chunks (1,250 rows) are fixed and aligned, so every strategy runs
  identical GEMMs. Distances use ‖a‖² + ‖b‖² − 2ab. Neighbours are the k smallest (distance, train
  index) pairs in lexicographic order, so ties are deterministic and the train-partitioned
  merge of local top-k lists equals the global top-k exactly. Majority vote over k=5 (vote ties
  go to the lowest label).
- **Preprocessing:** HOG feature matrices `np.allclose` to sequential.

If any check fails, `run_all.py` stops with exit code 2 and prints the failing check.

### Implementation notes

- Standardisation in S03 is per image (zero mean and unit variance for each image). This keeps the
  stage independent of batching, so it can stream through the pipeline. HOG uses 9 orientations,
  7×7-pixel cells and 2×2-cell blocks, giving 324 features per image.
- S03 loading reads 1000-image `.npz` shards (`data/shards/`). Stage-parallel workers load disjoint
  shards, and the pipeline loader streams them one batch at a time.
- The pipeline replicates only the HOG stage (`w` replicas). Load, normalise and standardise run
  as one process each, so a pipeline run uses `w + 3` processes in total. `Resource_Count` = `w`.
- K-means converges when the squared centroid shift is ≤ tol × mean feature variance
  (tol = 1e-4, max_iter = 50, the same rule as scikit-learn).

## Outputs

- `results/raw.csv`: one row per timed run, in the 19 required columns (Speedup and Efficiency left blank).
- `results/derived.csv`: raw rows plus Speedup, Efficiency, the baseline used, and overhead share.
- `results/config_stats.csv`: mean, median and std per configuration.
- `results/summary_table.csv` and `.md`: best time, best speedup, and speedup/efficiency at 8 workers.
- `results/s01_kmeans.csv` (iterations and inertia), `results/s02_knn.csv` (strategy, query count, queries/s, accuracy),
  `results/s03_stages.csv` (per-stage times).
- `results/correctness.csv`, `results/env.txt` (includes the pre-flight log), `results/run_log.txt`,
  `results/findings.md`, `results/sampler_diagnostic.csv`.
- `plots/`: the 9 required graphs, plus `s02_strategy_comparison`, `speedup_e2e_vs_compute` and
  `overhead_breakdown`. Each is saved as PNG (300 dpi, 6.5 in wide) and SVG.

## Report

- `report/CS10_23BCE0722_Report.docx`: the submitted case-study report (`.pdf` alongside it).
- `report/build_report.js`: builds the `.docx` from the results and figures (`npm install docx`, then `node report/build_report.js`).
- `report/make_figures.py`: the method diagrams and console-output figures in `report/figures/`.
- `report/table_data.py`: extracts the numbers used in the report tables into `report/table_data.json`.

`data/` (MNIST download and shards) is not committed; `python -m common.data` recreates it.
