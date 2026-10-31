# Findings (CS10, 23BCE0722): auto-generated facts

Baseline for every speedup: pure sequential implementation (no Pool, BLAS threads=1), mean of 5 timed runs at the same problem size. Pool/process start-up is INCLUDED in end-to-end Execution_Time (as Scheduling_Time); compute-only speedup = sequential / (Execution_Time - Scheduling_Time).

Environment: Apple M1; 8; 4 (Performance); 4 (Efficiency); 8.0 GiB
Memory sampler       : mode=track, 20 ms interval (see harness.MemSampler)
Memory sampler       : "track" mode for the final sweep (user decision); the original "scan" sampler

Correctness: 1854/1854 checks passed (accuracy_equal, centroids_allclose, features_allclose, iterations_equal, labels_identical, predictions_identical); warm-up runs and sequential-chunked runs are checked too.

Run-to-run variation: median CV 0.6%, max CV 16.9% (S02_kNN MNIST-q1000 kNN-brute-k5/train-partitioned n=30000 train-partitioned w=4).

## S01 Parallel K-means

**Blobs64d, KMeans-Lloyd-k10, n=10000**: sequential 0.034 s (std 0.000)
- multiprocessing: speedup w1=0.25x, w2=0.26x, w4=0.24x, w8=0.16x | compute-only w1=0.87x, w2=1.37x, w4=1.66x, w8=0.85x; efficiency@8=0.02, overhead share@8=86%; best at w=2; flattens after w=1 (w2/w1 = 1.04); w=1 is 3.95x slower than pure sequential
  - w8 overheads (mean s): comm 0.004, sync 0.007, sched 0.168 of 0.208 total; peak RSS 1192 MB

**Blobs64d, KMeans-Lloyd-k10, n=30000**: sequential 0.108 s (std 0.000)
- multiprocessing: speedup w1=0.52x, w2=0.63x, w4=0.66x, w8=0.46x | compute-only w1=0.94x, w2=1.54x, w4=2.21x, w8=1.62x; efficiency@8=0.06, overhead share@8=76%; best at w=4; flattens after w=2 (w4/w2 = 1.04); w=1 is 1.92x slower than pure sequential
  - w8 overheads (mean s): comm 0.005, sync 0.007, sched 0.165 of 0.233 total; peak RSS 1264 MB

**Blobs64d, KMeans-Lloyd-k10, n=60000**: sequential 0.160 s (std 0.000)
- multiprocessing: speedup w1=0.62x, w2=0.80x, w4=0.88x, w8=0.66x | compute-only w1=0.94x, w2=1.53x, w4=2.22x, w8=1.92x; efficiency@8=0.08, overhead share@8=71%; best at w=4; flattens after w=4 (w8/w4 = 0.75); w=1 is 1.61x slower than pure sequential
  - w8 overheads (mean s): comm 0.004, sync 0.008, sched 0.160 of 0.244 total; peak RSS 934 MB

**MNIST, KMeans-Lloyd-k10, n=10000**: sequential 0.414 s (std 0.002)
- multiprocessing: speedup w1=0.79x, w2=1.15x, w4=1.33x, w8=1.09x | compute-only w1=0.94x, w2=1.57x, w4=2.06x, w8=1.79x; efficiency@8=0.14, overhead share@8=47%; best at w=4; flattens after w=4 (w8/w4 = 0.82); w=1 is 1.27x slower than pure sequential
  - w8 overheads (mean s): comm 0.011, sync 0.020, sched 0.148 of 0.380 total; peak RSS 1016 MB

**MNIST, KMeans-Lloyd-k10, n=30000**: sequential 1.148 s (std 0.000)
- multiprocessing: speedup w1=0.91x, w2=1.48x, w4=1.88x, w8=1.82x | compute-only w1=0.96x, w2=1.63x, w4=2.14x, w8=2.19x; efficiency@8=0.23, overhead share@8=22%; best at w=4; flattens after w=4 (w8/w4 = 0.97); w=1 is 1.10x slower than pure sequential
  - w8 overheads (mean s): comm 0.015, sync 0.019, sched 0.107 of 0.632 total; peak RSS 1747 MB

**MNIST, KMeans-Lloyd-k20, n=30000**: sequential 1.392 s (std 0.002)
- multiprocessing: speedup w1=0.92x, w2=1.45x, w4=1.86x, w8=1.82x | compute-only w1=0.97x, w2=1.57x, w4=2.09x, w8=2.11x; efficiency@8=0.23, overhead share@8=19%; best at w=4; flattens after w=4 (w8/w4 = 0.98); w=1 is 1.08x slower than pure sequential
  - w8 overheads (mean s): comm 0.019, sync 0.026, sched 0.103 of 0.763 total; peak RSS 1688 MB

**MNIST, KMeans-Lloyd-k5, n=30000**: sequential 0.879 s (std 0.006)
- multiprocessing: speedup w1=0.89x, w2=1.43x, w4=1.77x, w8=1.65x | compute-only w1=0.95x, w2=1.61x, w4=2.07x, w8=2.04x; efficiency@8=0.21, overhead share@8=24%; best at w=4; flattens after w=4 (w8/w4 = 0.93); w=1 is 1.12x slower than pure sequential
  - w8 overheads (mean s): comm 0.013, sync 0.015, sched 0.101 of 0.532 total; peak RSS 1635 MB

**MNIST, KMeans-Lloyd-k10, n=60000**: sequential 2.662 s (std 0.004)
- multiprocessing: speedup w1=0.95x, w2=1.63x, w4=2.22x, w8=2.29x | compute-only w1=0.96x, w2=1.67x, w4=2.30x, w8=2.44x; efficiency@8=0.29, overhead share@8=10%; best at w=8; flattens after w=4 (w8/w4 = 1.03); w=1 is 1.05x slower than pure sequential
  - w8 overheads (mean s): comm 0.024, sync 0.022, sched 0.070 of 1.161 total; peak RSS 2493 MB

Iterations to convergence (identical for sequential, sequential-chunked and all worker counts):
- Blobs64d n=10000 k=10: 22 iterations
- Blobs64d n=30000 k=10: 23 iterations
- Blobs64d n=60000 k=10: 17 iterations
- MNIST n=10000 k=10: 46 iterations
- MNIST n=30000 k=5: 33 iterations
- MNIST n=30000 k=10: 43 iterations
- MNIST n=30000 k=20: 50 iterations (hit max_iter=50, not converged)
- MNIST n=60000 k=10: 50 iterations (hit max_iter=50, not converged)

Anomalies: 
- Blobs64d KMeans-Lloyd-k10 n=10000 multiprocessing: w4 slower than w2 (0.24x < 0.26x)
- Blobs64d KMeans-Lloyd-k10 n=10000 multiprocessing: w8 slower than w4 (0.16x < 0.24x)
- Blobs64d KMeans-Lloyd-k10 n=30000 multiprocessing: w8 slower than w4 (0.46x < 0.66x)
- Blobs64d KMeans-Lloyd-k10 n=60000 multiprocessing: w8 slower than w4 (0.66x < 0.88x)
- MNIST KMeans-Lloyd-k10 n=10000 multiprocessing: w8 slower than w4 (1.09x < 1.33x)
- MNIST KMeans-Lloyd-k10 n=30000 multiprocessing: w8 slower than w4 (1.82x < 1.88x)
- MNIST KMeans-Lloyd-k20 n=30000 multiprocessing: w8 slower than w4 (1.82x < 1.86x)
- MNIST KMeans-Lloyd-k5 n=30000 multiprocessing: w8 slower than w4 (1.65x < 1.77x)

## S02 Parallel k-NN

**MNIST-q1000, kNN-brute-k5, n=10000**: sequential 0.122 s (std 0.001)
- query-partitioned: speedup w1=0.54x, w2=0.67x, w4=0.68x, w8=0.52x | compute-only w1=0.94x, w2=1.56x, w4=2.08x, w8=1.86x; efficiency@8=0.06, overhead share@8=77%; best at w=4; flattens after w=2 (w4/w2 = 1.02); w=1 is 1.84x slower than pure sequential
  - w8 overheads (mean s): comm 0.003, sync 0.007, sched 0.171 of 0.236 total; peak RSS 1461 MB
- train-partitioned: speedup w1=0.54x, w2=0.66x, w4=0.67x, w8=0.51x | compute-only w1=0.94x, w2=1.54x, w4=1.95x, w8=1.80x; efficiency@8=0.06, overhead share@8=74%; best at w=4; flattens after w=2 (w4/w2 = 1.01); w=1 is 1.84x slower than pure sequential
  - w8 overheads (mean s): comm 0.003, sync 0.004, sched 0.169 of 0.237 total; peak RSS 1229 MB

**MNIST-q1000, kNN-brute-k5, n=30000**: sequential 0.334 s (std 0.001)
- query-partitioned: speedup w1=0.75x, w2=1.03x, w4=1.15x, w8=0.99x | compute-only w1=0.96x, w2=1.52x, w4=1.94x, w8=2.04x; efficiency@8=0.12, overhead share@8=62%; best at w=4; flattens after w=4 (w8/w4 = 0.86); w=1 is 1.33x slower than pure sequential
  - w8 overheads (mean s): comm 0.008, sync 0.028, sched 0.173 of 0.337 total; peak RSS 2115 MB
- train-partitioned: speedup w1=0.74x, w2=1.01x, w4=1.16x, w8=1.05x | compute-only w1=0.95x, w2=1.46x, w4=2.03x, w8=2.18x; efficiency@8=0.13, overhead share@8=62%; best at w=4; flattens after w=4 (w8/w4 = 0.90); w=1 is 1.34x slower than pure sequential
  - w8 overheads (mean s): comm 0.008, sync 0.025, sched 0.166 of 0.320 total; peak RSS 1192 MB

**MNIST-q1000, kNN-brute-k5, n=60000**: sequential 0.712 s (std 0.002)
- query-partitioned: speedup w1=0.84x, w2=1.26x, w4=1.52x, w8=1.44x | compute-only w1=0.95x, w2=1.54x, w4=2.05x, w8=2.22x; efficiency@8=0.18, overhead share@8=45%; best at w=4; flattens after w=4 (w8/w4 = 0.95); w=1 is 1.19x slower than pure sequential
  - w8 overheads (mean s): comm 0.016, sync 0.033, sched 0.173 of 0.494 total; peak RSS 3168 MB
- train-partitioned: speedup w1=0.84x, w2=1.37x, w4=1.70x, w8=1.72x | compute-only w1=0.95x, w2=1.70x, w4=2.39x, w8=2.90x; efficiency@8=0.21, overhead share@8=51%; best at w=8; flattens after w=4 (w8/w4 = 1.01); w=1 is 1.19x slower than pure sequential
  - w8 overheads (mean s): comm 0.015, sync 0.029, sched 0.168 of 0.414 total; peak RSS 1276 MB

**MNIST-q10000, kNN-brute-k5, n=10000**: sequential 1.208 s (std 0.001)
- query-partitioned: speedup w1=0.92x, w2=1.54x, w4=2.16x, w8=2.35x | compute-only w1=0.99x, w2=1.78x, w4=2.75x, w8=3.44x; efficiency@8=0.29, overhead share@8=38%; best at w=8; flattens after w=4 (w8/w4 = 1.09); w=1 is 1.09x slower than pure sequential
  - w8 overheads (mean s): comm 0.005, sync 0.028, sched 0.163 of 0.515 total; peak RSS 1264 MB
- train-partitioned: speedup w1=0.92x, w2=1.48x, w4=1.85x, w8=2.08x | compute-only w1=0.99x, w2=1.70x, w4=2.27x, w8=2.93x; efficiency@8=0.26, overhead share@8=34%; best at w=8; w=1 is 1.09x slower than pure sequential
  - w8 overheads (mean s): comm 0.005, sync 0.026, sched 0.169 of 0.581 total; peak RSS 1171 MB

**MNIST-q10000, kNN-brute-k5, n=30000**: sequential 3.362 s (std 0.064)
- query-partitioned: speedup w1=0.98x, w2=1.71x, w4=2.29x, w8=2.90x | compute-only w1=1.00x, w2=1.81x, w4=2.49x, w8=3.41x; efficiency@8=0.36, overhead share@8=18%; best at w=8; w=1 is 1.03x slower than pure sequential
  - w8 overheads (mean s): comm 0.010, sync 0.021, sched 0.174 of 1.159 total; peak RSS 1825 MB
- train-partitioned: speedup w1=0.97x, w2=1.54x, w4=2.41x, w8=2.71x | compute-only w1=1.00x, w2=1.62x, w4=2.64x, w8=3.14x; efficiency@8=0.34, overhead share@8=16%; best at w=8; w=1 is 1.03x slower than pure sequential
  - w8 overheads (mean s): comm 0.010, sync 0.015, sched 0.171 of 1.243 total; peak RSS 1109 MB

**MNIST-q10000, kNN-brute-k5, n=60000**: sequential 6.884 s (std 0.004)
- query-partitioned: speedup w1=0.98x, w2=1.71x, w4=2.55x, w8=3.06x | compute-only w1=0.99x, w2=1.76x, w4=2.67x, w8=3.32x; efficiency@8=0.38, overhead share@8=10%; best at w=8; w=1 is 1.02x slower than pure sequential
  - w8 overheads (mean s): comm 0.018, sync 0.025, sched 0.176 of 2.249 total; peak RSS 3112 MB
- train-partitioned: speedup w1=0.98x, w2=1.73x, w4=2.58x, w8=3.29x | compute-only w1=0.99x, w2=1.78x, w4=2.71x, w8=3.58x; efficiency@8=0.41, overhead share@8=11%; best at w=8; w=1 is 1.02x slower than pure sequential
  - w8 overheads (mean s): comm 0.018, sync 0.043, sched 0.167 of 2.091 total; peak RSS 1407 MB

**MNIST-q5000, kNN-brute-k5, n=10000**: sequential 0.608 s (std 0.000)
- query-partitioned: speedup w1=0.85x, w2=1.32x, w4=1.70x, w8=1.67x | compute-only w1=0.99x, w2=1.71x, w4=2.55x, w8=3.16x; efficiency@8=0.21, overhead share@8=55%; best at w=4; flattens after w=4 (w8/w4 = 0.98); w=1 is 1.17x slower than pure sequential
  - w8 overheads (mean s): comm 0.004, sync 0.025, sched 0.170 of 0.364 total; peak RSS 1375 MB
- train-partitioned: speedup w1=0.86x, w2=1.31x, w4=1.56x, w8=1.56x | compute-only w1=0.99x, w2=1.68x, w4=2.24x, w8=2.75x; efficiency@8=0.20, overhead share@8=50%; best at w=8; flattens after w=4 (w8/w4 = 1.00); w=1 is 1.17x slower than pure sequential
  - w8 overheads (mean s): comm 0.004, sync 0.024, sched 0.168 of 0.390 total; peak RSS 981 MB

**MNIST-q5000, kNN-brute-k5, n=30000**: sequential 1.660 s (std 0.001)
- query-partitioned: speedup w1=0.93x, w2=1.50x, w4=2.03x, w8=2.38x | compute-only w1=0.99x, w2=1.66x, w4=2.38x, w8=3.14x; efficiency@8=0.30, overhead share@8=30%; best at w=8; w=1 is 1.07x slower than pure sequential
  - w8 overheads (mean s): comm 0.009, sync 0.030, sched 0.169 of 0.698 total; peak RSS 1967 MB
- train-partitioned: speedup w1=0.94x, w2=1.44x, w4=2.14x, w8=1.88x | compute-only w1=0.99x, w2=1.58x, w4=2.53x, w8=2.36x; efficiency@8=0.24, overhead share@8=28%; best at w=4; flattens after w=4 (w8/w4 = 0.88); w=1 is 1.07x slower than pure sequential
  - w8 overheads (mean s): comm 0.010, sync 0.061, sched 0.177 of 0.883 total; peak RSS 1086 MB

**MNIST-q5000, kNN-brute-k5, n=60000**: sequential 3.540 s (std 0.002)
- query-partitioned: speedup w1=0.97x, w2=1.62x, w4=2.18x, w8=2.72x | compute-only w1=0.99x, w2=1.71x, w4=2.36x, w8=3.14x; efficiency@8=0.34, overhead share@8=17%; best at w=8; w=1 is 1.04x slower than pure sequential
  - w8 overheads (mean s): comm 0.016, sync 0.025, sched 0.174 of 1.303 total; peak RSS 3106 MB
- train-partitioned: speedup w1=0.96x, w2=1.72x, w4=2.57x, w8=3.04x | compute-only w1=0.99x, w2=1.81x, w4=2.81x, w8=3.54x; efficiency@8=0.38, overhead share@8=19%; best at w=8; w=1 is 1.04x slower than pure sequential
  - w8 overheads (mean s): comm 0.016, sync 0.037, sched 0.166 of 1.166 total; peak RSS 1234 MB

Accuracy (first N MNIST test images as queries; identical for every configuration and strategy):
- train n=10000, 1000 queries: 0.9160
- train n=10000, 5000 queries: 0.9222
- train n=10000, 10000 queries: 0.9442
- train n=30000, 1000 queries: 0.9470
- train n=30000, 5000 queries: 0.9456
- train n=30000, 10000 queries: 0.9627
- train n=60000, 1000 queries: 0.9610
- train n=60000, 5000 queries: 0.9546
- train n=60000, 10000 queries: 0.9688
Peak throughput (queries/s) per train size x query count:
- n=10000, 1000 q: 8214 (none w=1); sequential 8214
- n=10000, 5000 q: 14010 (query-partitioned w=4); sequential 8217
- n=10000, 10000 q: 19416 (query-partitioned w=8); sequential 8280
- n=30000, 1000 q: 3480 (train-partitioned w=4); sequential 2990
- n=30000, 5000 q: 7163 (query-partitioned w=8); sequential 3013
- n=30000, 10000 q: 8631 (query-partitioned w=8); sequential 2975
- n=60000, 1000 q: 2414 (train-partitioned w=8); sequential 1405
- n=60000, 5000 q: 4289 (train-partitioned w=8); sequential 1412
- n=60000, 10000 q: 4785 (train-partitioned w=8); sequential 1453
Strategy comparison at w=8 (end-to-end speedup, query-partitioned vs train-partitioned):
- MNIST-q1000 n=10000: query 0.52x vs train 0.51x
- MNIST-q1000 n=30000: query 0.99x vs train 1.05x
- MNIST-q1000 n=60000: query 1.44x vs train 1.72x
- MNIST-q10000 n=10000: query 2.35x vs train 2.08x
- MNIST-q10000 n=30000: query 2.90x vs train 2.71x
- MNIST-q10000 n=60000: query 3.06x vs train 3.29x
- MNIST-q5000 n=10000: query 1.67x vs train 1.56x
- MNIST-q5000 n=30000: query 2.38x vs train 1.88x
- MNIST-q5000 n=60000: query 2.72x vs train 3.04x

Anomalies: 
- MNIST-q1000 kNN-brute-k5 n=10000 query-partitioned: w8 slower than w4 (0.52x < 0.68x)
- MNIST-q1000 kNN-brute-k5 n=10000 train-partitioned: w8 slower than w4 (0.51x < 0.67x)
- MNIST-q1000 kNN-brute-k5 n=30000 query-partitioned: w8 slower than w4 (0.99x < 1.15x)
- MNIST-q1000 kNN-brute-k5 n=30000 train-partitioned: w8 slower than w4 (1.05x < 1.16x)
- MNIST-q1000 kNN-brute-k5 n=60000 query-partitioned: w8 slower than w4 (1.44x < 1.52x)
- MNIST-q5000 kNN-brute-k5 n=10000 query-partitioned: w8 slower than w4 (1.67x < 1.70x)
- MNIST-q5000 kNN-brute-k5 n=30000 train-partitioned: w8 slower than w4 (1.88x < 2.14x)

## S03 Parallel preprocessing pipeline

**MNIST, load+norm+std+HOG, n=10000**: sequential 0.876 s (std 0.005)
- pipeline: speedup w1=0.88x, w2=1.45x, w4=1.80x, w8=1.73x | compute-only w1=0.90x, w2=1.50x, w4=1.90x, w8=1.87x; efficiency@8=0.22, overhead share@8=19%; best at w=4; flattens after w=4 (w8/w4 = 0.96); w=1 is 1.14x slower than pure sequential
  - w8 overheads (mean s): comm 0.025, sync 0.033, sched 0.037 of 0.505 total; peak RSS 1046 MB
- stage-parallel: speedup w1=0.89x, w2=1.55x, w4=2.19x, w8=2.08x | compute-only w1=0.99x, w2=1.91x, w4=3.18x, w8=3.66x; efficiency@8=0.26, overhead share@8=48%; best at w=4; flattens after w=4 (w8/w4 = 0.95); w=1 is 1.13x slower than pure sequential
  - w8 overheads (mean s): comm 0.005, sync 0.014, sched 0.182 of 0.422 total; peak RSS 976 MB

**MNIST, load+norm+std+HOG, n=30000**: sequential 2.616 s (std 0.008)
- pipeline: speedup w1=0.95x, w2=1.74x, w4=2.56x, w8=2.87x | compute-only w1=0.95x, w2=1.77x, w4=2.62x, w8=3.01x; efficiency@8=0.36, overhead share@8=19%; best at w=8; w=1 is 1.05x slower than pure sequential
  - w8 overheads (mean s): comm 0.071, sync 0.058, sched 0.043 of 0.913 total; peak RSS 1737 MB
- stage-parallel: speedup w1=0.95x, w2=1.77x, w4=2.83x, w8=3.07x | compute-only w1=0.99x, w2=1.91x, w4=3.27x, w8=3.89x; efficiency@8=0.38, overhead share@8=24%; best at w=8; flattens after w=4 (w8/w4 = 1.09); w=1 is 1.05x slower than pure sequential
  - w8 overheads (mean s): comm 0.013, sync 0.013, sched 0.178 of 0.851 total; peak RSS 1814 MB

**MNIST, load+norm+std+HOG, n=60000**: sequential 5.261 s (std 0.036)
- pipeline: speedup w1=0.98x, w2=1.83x, w4=2.84x, w8=3.28x | compute-only w1=0.98x, w2=1.84x, w4=2.89x, w8=3.37x; efficiency@8=0.41, overhead share@8=18%; best at w=8; w=1 is 1.02x slower than pure sequential
  - w8 overheads (mean s): comm 0.152, sync 0.089, sched 0.042 of 1.606 total; peak RSS 2253 MB
- stage-parallel: speedup w1=0.96x, w2=1.83x, w4=2.99x, w8=3.47x | compute-only w1=0.99x, w2=1.91x, w4=3.23x, w8=3.94x; efficiency@8=0.43, overhead share@8=15%; best at w=8; w=1 is 1.04x slower than pure sequential
  - w8 overheads (mean s): comm 0.026, sync 0.017, sched 0.181 of 1.518 total; peak RSS 2488 MB

Stage shares, sequential (mean wall time):
- n=10000: load 0.003s (0.3%), normalise 0.003s (0.4%), standardise 0.010s (1.1%), hog 0.860s (98.2%); bottleneck = hog
- n=30000: load 0.008s (0.3%), normalise 0.010s (0.4%), standardise 0.038s (1.5%), hog 2.559s (97.8%); bottleneck = hog
- n=60000: load 0.022s (0.4%), normalise 0.020s (0.4%), standardise 0.099s (1.9%), hog 5.120s (97.3%); bottleneck = hog
Stage-parallel per-stage wall time at n=60000 (s), vs sequential:
- load: seq 0.022; w1 0.131, w2 0.106, w4 0.108, w8 0.146; stage speedup@8 0.15x
- normalise: seq 0.020; w1 0.032, w2 0.018, w4 0.011, w8 0.017; stage speedup@8 1.21x
- standardise: seq 0.099; w1 0.080, w2 0.049, w4 0.042, w8 0.041; stage speedup@8 2.39x
- hog: seq 5.120; w1 5.161, w2 2.646, w4 1.535, w8 1.223; stage speedup@8 4.19x
Pipeline busy CPU time per stage at n=60000 (s, summed over replicas):
- w1: load 0.034, normalise 0.023, standardise 0.063, hog 5.140
- w2: load 0.048, normalise 0.026, standardise 0.065, hog 5.338
- w4: load 0.062, normalise 0.050, standardise 0.133, hog 6.460
- w8: load 0.073, normalise 0.059, standardise 0.157, hog 9.967
- HOG busy CPU time inflates 1.94x from w1 to w8 (same work, so per-image cost rises with 8 processes on 4P+4E cores)
Throughput at n=60000 (images/s): pipeline w1: 11170; pipeline w2: 20832; pipeline w4: 32434; pipeline w8: 37371; sequential w1: 11404; sequential-chunked w1: 11517; stage-parallel w1: 10991; stage-parallel w2: 20859; stage-parallel w4: 34106; stage-parallel w8: 39518

Anomalies: 
- MNIST load+norm+std+HOG n=10000 pipeline: w8 slower than w4 (1.73x < 1.80x)
- MNIST load+norm+std+HOG n=10000 stage-parallel: w8 slower than w4 (2.08x < 2.19x)

## Check: is w=1 > sequential explained by chunking (cache locality)?

sequential-chunked = the w=8 task decomposition (same chunk ranges, same task kernel, same reduction order) executed in-process with no Pool. If chunking/cache locality caused w=1 > sequential, sequential-chunked should be faster than sequential by a similar margin. Note: at w=1 a worker's chunk is the whole problem, i.e. the same chunking as sequential.

| Workload | Dataset | n | seq (s) | seq-chunked (s) | chunked gain | w=1 variant | w=1 (s) | w=1 gain | verdict |
|---|---|---|---|---|---|---|---|---|---|
| S01_KMeans | Blobs64d | 10000 | 0.034 | 0.034 | -1.6% | multiprocessing | 0.132 | -74.7% | no w=1 anomaly |
| S01_KMeans | Blobs64d | 30000 | 0.108 | 0.109 | -0.7% | multiprocessing | 0.209 | -48.0% | no w=1 anomaly |
| S01_KMeans | Blobs64d | 60000 | 0.160 | 0.161 | -0.3% | multiprocessing | 0.259 | -38.0% | no w=1 anomaly |
| S01_KMeans | MNIST | 10000 | 0.414 | 0.414 | -0.2% | multiprocessing | 0.525 | -21.2% | no w=1 anomaly |
| S01_KMeans | MNIST | 30000 | 1.148 | 1.161 | -1.1% | multiprocessing | 1.260 | -8.9% | no w=1 anomaly |
| S01_KMeans | MNIST | 30000 | 1.392 | 1.397 | -0.4% | multiprocessing | 1.506 | -7.5% | no w=1 anomaly |
| S01_KMeans | MNIST | 30000 | 0.879 | 0.878 | +0.1% | multiprocessing | 0.983 | -10.6% | no w=1 anomaly |
| S01_KMeans | MNIST | 60000 | 2.662 | 2.671 | -0.3% | multiprocessing | 2.807 | -5.2% | no w=1 anomaly |
| S02_kNN | MNIST-q1000 | 10000 | 0.122 | 0.121 | +0.3% | query-partitioned | 0.224 | -45.7% | no w=1 anomaly |
| S02_kNN | MNIST-q1000 | 10000 | 0.122 | 0.121 | +0.3% | train-partitioned | 0.225 | -45.8% | no w=1 anomaly |
| S02_kNN | MNIST-q1000 | 30000 | 0.334 | 0.334 | +0.1% | query-partitioned | 0.443 | -24.5% | no w=1 anomaly |
| S02_kNN | MNIST-q1000 | 30000 | 0.334 | 0.334 | +0.1% | train-partitioned | 0.449 | -25.6% | no w=1 anomaly |
| S02_kNN | MNIST-q1000 | 60000 | 0.712 | 0.721 | -1.3% | query-partitioned | 0.845 | -15.7% | no w=1 anomaly |
| S02_kNN | MNIST-q1000 | 60000 | 0.712 | 0.721 | -1.3% | train-partitioned | 0.845 | -15.8% | no w=1 anomaly |
| S02_kNN | MNIST-q10000 | 10000 | 1.208 | 1.210 | -0.2% | query-partitioned | 1.314 | -8.1% | no w=1 anomaly |
| S02_kNN | MNIST-q10000 | 10000 | 1.208 | 1.210 | -0.2% | train-partitioned | 1.316 | -8.2% | no w=1 anomaly |
| S02_kNN | MNIST-q10000 | 30000 | 3.362 | 3.329 | +1.0% | query-partitioned | 3.448 | -2.5% | no w=1 anomaly |
| S02_kNN | MNIST-q10000 | 30000 | 3.362 | 3.329 | +1.0% | train-partitioned | 3.449 | -2.5% | no w=1 anomaly |
| S02_kNN | MNIST-q10000 | 60000 | 6.884 | 6.886 | -0.0% | query-partitioned | 7.014 | -1.9% | no w=1 anomaly |
| S02_kNN | MNIST-q10000 | 60000 | 6.884 | 6.886 | -0.0% | train-partitioned | 7.037 | -2.2% | no w=1 anomaly |
| S02_kNN | MNIST-q5000 | 10000 | 0.608 | 0.609 | -0.0% | query-partitioned | 0.712 | -14.5% | no w=1 anomaly |
| S02_kNN | MNIST-q5000 | 10000 | 0.608 | 0.609 | -0.0% | train-partitioned | 0.711 | -14.4% | no w=1 anomaly |
| S02_kNN | MNIST-q5000 | 30000 | 1.660 | 1.658 | +0.1% | query-partitioned | 1.782 | -6.8% | no w=1 anomaly |
| S02_kNN | MNIST-q5000 | 30000 | 1.660 | 1.658 | +0.1% | train-partitioned | 1.773 | -6.4% | no w=1 anomaly |
| S02_kNN | MNIST-q5000 | 60000 | 3.540 | 3.540 | -0.0% | query-partitioned | 3.666 | -3.4% | no w=1 anomaly |
| S02_kNN | MNIST-q5000 | 60000 | 3.540 | 3.540 | -0.0% | train-partitioned | 3.671 | -3.5% | no w=1 anomaly |
| S03_Preprocess | MNIST | 10000 | 0.876 | 0.869 | +0.8% | pipeline | 0.995 | -12.0% | no w=1 anomaly |
| S03_Preprocess | MNIST | 10000 | 0.876 | 0.869 | +0.8% | stage-parallel | 0.986 | -11.1% | no w=1 anomaly |
| S03_Preprocess | MNIST | 30000 | 2.616 | 2.609 | +0.3% | pipeline | 2.759 | -5.2% | no w=1 anomaly |
| S03_Preprocess | MNIST | 30000 | 2.616 | 2.609 | +0.3% | stage-parallel | 2.748 | -4.8% | no w=1 anomaly |
| S03_Preprocess | MNIST | 60000 | 5.261 | 5.210 | +1.0% | pipeline | 5.372 | -2.1% | no w=1 anomaly |
| S03_Preprocess | MNIST | 60000 | 5.261 | 5.210 | +1.0% | stage-parallel | 5.459 | -3.6% | no w=1 anomaly |

Verdict counts: no w=1 anomaly: 32
(gain = sequential time / variant time - 1; noise threshold = max(2%, 2x CV of either baseline))

## Diagnostic: memory-sampler cost on in-process (sequential) runs

Pure sequential runs timed with the memory sampler in each mode (analysis/sampler_diagnostic.py). 'scan' = original sampler (psutil children(recursive=True) every 20 ms, scans every process on the system under the parent's GIL); 'track' = only registered worker PIDs; 'off' = none.

| Case | Mode | mean (s) | std (s) | runs | vs off |
|---|---|---|---|---|---|
| S01 K-means MNIST k=10 | off | 2.661 | 0.003 | 3 | +0.0% |
| S01 K-means MNIST k=10 | scan | 2.954 | 0.002 | 3 | +11.0% |
| S01 K-means MNIST k=10 | track | 2.666 | 0.004 | 3 | +0.2% |
| S02 k-NN 1000 queries | off | 0.701 | 0.001 | 3 | +0.0% |
| S02 k-NN 1000 queries | scan | 0.732 | 0.008 | 3 | +4.5% |
| S02 k-NN 1000 queries | track | 0.707 | 0.007 | 3 | +0.9% |
| S03 preprocessing | off | 5.209 | 0.022 | 3 | +0.0% |
| S03 preprocessing | scan | 5.729 | 0.048 | 3 | +10.0% |
| S03 preprocessing | track | 5.196 | 0.027 | 3 | -0.3% |

