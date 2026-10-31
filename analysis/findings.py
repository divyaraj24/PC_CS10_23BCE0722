"""Write results/findings.md: key numbers + anomalies per sub-problem (facts only)."""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
W = [1, 2, 4, 8]
FLAT = 1.10  # doubling workers gives <10% more speedup => "flattens"
BKEY = ["Workload", "Dataset", "Problem_Size", "Base_Algorithm"]
PAR_EXCLUDE = ("sequential", "sequential-chunked")


def fmt(x, p=2):
    return f"{x:.{p}f}"


def scaling(gm, label):
    s = gm.set_index("Resource_Count")
    sp = {w: s.loc[w, "Speedup_Mean"] for w in W if w in s.index}
    cs = {w: s.loc[w, "Compute_Speedup_Mean"] for w in W if w in s.index}
    line = (f"- {label}: speedup " + ", ".join(f"w{w}={fmt(v)}x" for w, v in sp.items())
            + " | compute-only " + ", ".join(f"w{w}={fmt(v)}x" for w, v in cs.items()))
    notes, anomalies = [], []
    if 8 in s.index:
        notes.append(f"efficiency@8={fmt(s.loc[8, 'Efficiency_Mean'])}, overhead share@8={s.loc[8, 'Overhead_Share_Mean']:.0%}")
    best = max(sp, key=sp.get)
    notes.append(f"best at w={best}")
    flat = next((w for w in W[:-1] if w in sp and 2 * w in sp and sp[2 * w] / sp[w] < FLAT), None)
    if flat:
        notes.append(f"flattens after w={flat} (w{2 * flat}/w{flat} = {sp[2 * flat] / sp[flat]:.2f})")
    if sp.get(1, 1) < 1:
        notes.append(f"w=1 is {1 / sp[1]:.2f}x slower than pure sequential")
    anomalies += [f"w{b} slower than w{a} ({fmt(sp[b])}x < {fmt(sp[a])}x)"
                  for a, b in zip(W, W[1:]) if a in sp and b in sp and sp[b] < sp[a]]
    if sp.get(1, 1) > 1.03:
        anomalies.append(f"w=1 faster than pure sequential ({fmt(sp[1])}x)")
    lines = [line + "; " + "; ".join(notes)]
    if 8 in s.index:
        r = s.loc[8]
        lines.append(f"  - w8 overheads (mean s): comm {fmt(r.Communication_Time_Mean, 3)}, sync "
                     f"{fmt(r.Synchronization_Time_Mean, 3)}, sched {fmt(r.Scheduling_Time_Mean, 3)} of "
                     f"{fmt(r.Time_Mean, 3)} total; peak RSS {r.Memory_Use_Mean:.0f} MB")
    return lines, anomalies


def section(cs, workload, title, extra=None):
    lines, anomalies = [f"## {title}", ""], []
    for (_, ds, n, alg), g in cs[cs.Workload == workload].groupby(BKEY):
        seq = g[g.Parallel_Model == "sequential"].iloc[0]
        lines.append(f"**{ds}, {alg}, n={n}**: sequential {fmt(seq.Time_Mean, 3)} s (std {fmt(seq.Time_Std, 3)})")
        for var, gm in g[~g.Parallel_Model.isin(PAR_EXCLUDE)].groupby("Variant"):
            l, an = scaling(gm, var)
            lines += l
            anomalies += [f"{ds} {alg} n={n} {var}: {x}" for x in an]
        lines.append("")
    if extra:
        lines += extra + [""]
    lines.append("Anomalies: " + ("none detected" if not anomalies else ""))
    lines += [f"- {a}" for a in anomalies]
    return lines + [""], anomalies


def locality_section(cs):
    """sequential vs sequential-chunked vs w=1, per configuration."""
    L = ["## Check: is w=1 > sequential explained by chunking (cache locality)?", "",
         "sequential-chunked = the w=8 task decomposition (same chunk ranges, same task kernel, same "
         "reduction order) executed in-process with no Pool. If chunking/cache locality caused w=1 > "
         "sequential, sequential-chunked should be faster than sequential by a similar margin. "
         "Note: at w=1 a worker's chunk is the whole problem, i.e. the same chunking as sequential.", "",
         "| Workload | Dataset | n | seq (s) | seq-chunked (s) | chunked gain | w=1 variant | w=1 (s) | w=1 gain | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for (wl, ds, n, alg), g in cs.groupby(BKEY):
        seq = g[g.Parallel_Model == "sequential"]
        ch = g[g.Parallel_Model == "sequential-chunked"]
        if seq.empty or ch.empty:
            continue
        seq, ch = seq.iloc[0], ch.iloc[0]
        noise = max(0.02, 2 * seq.Time_Std / seq.Time_Mean, 2 * ch.Time_Std / ch.Time_Mean)
        cg = seq.Time_Mean / ch.Time_Mean - 1
        for var, w1 in g[(~g.Parallel_Model.isin(PAR_EXCLUDE)) & (g.Resource_Count == 1)].groupby("Variant"):
            w1 = w1.iloc[0]
            wg = seq.Time_Mean / w1.Time_Mean - 1
            if wg <= noise:
                v = "no w=1 anomaly"
            elif cg > noise and cg >= 0.5 * wg:
                v = "SUPPORTS chunking/locality"
            else:
                v = "does NOT support chunking/locality"
            verdicts.append(v)
            L.append(f"| {wl} | {ds} | {n} | {fmt(seq.Time_Mean, 3)} | {fmt(ch.Time_Mean, 3)} | {cg:+.1%} | "
                     f"{var} | {fmt(w1.Time_Mean, 3)} | {wg:+.1%} | {v} |")
    counts = pd.Series(verdicts).value_counts()
    L += ["", "Verdict counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()),
          "(gain = sequential time / variant time - 1; noise threshold = max(2%, 2x CV of either baseline))", ""]
    return L


def main(results):
    cs = pd.read_csv(results / "config_stats.csv")
    corr = pd.read_csv(results / "correctness.csv")
    d = pd.read_csv(results / "derived.csv")
    runs = int(cs.Runs.mode().iloc[0])
    L = ["# Findings (CS10, 23BCE0722): auto-generated facts", "",
         f"Baseline for every speedup: pure sequential implementation (no Pool, BLAS threads=1), mean of "
         f"{runs} timed runs at the same problem size. Pool/process start-up is INCLUDED in end-to-end "
         "Execution_Time (as Scheduling_Time); compute-only speedup = sequential / (Execution_Time - "
         "Scheduling_Time).", ""]
    env = (results / "env.txt").read_text().splitlines() if (results / "env.txt").exists() else []
    L += ["Environment: " + "; ".join(x.split(":", 1)[1].strip() for x in env[:5])]
    L += [x.strip() for x in env if x.startswith("Memory sampler")] + [""]
    passed = int(corr.Passed.astype(str).eq("True").sum())
    L += [f"Correctness: {passed}/{len(corr)} checks passed ({', '.join(sorted(corr.Check.unique()))}); "
          "warm-up runs and sequential-chunked runs are checked too.", ""]
    cv = cs.Time_Std / cs.Time_Mean
    worst = cs.loc[cv.idxmax()]
    L += [f"Run-to-run variation: median CV {cv.median():.1%}, max CV {cv.max():.1%} "
          f"({worst.Workload} {worst.Dataset} {worst.Algorithm} n={worst.Problem_Size} "
          f"{worst.Variant} w={worst.Resource_Count}).", ""]

    it = pd.read_csv(results / "s01_kmeans.csv")
    its = it[it.Parallel_Model == "sequential"].groupby(["Dataset", "Problem_Size", "k"]).agg(
        Iterations=("Iterations", "first"), Converged=("Converged", "first")).reset_index()
    ex1 = ["Iterations to convergence (identical for sequential, sequential-chunked and all worker counts):"]
    ex1 += [f"- {r.Dataset} n={r.Problem_Size} k={r.k}: {r.Iterations} iterations"
            + ("" if str(r.Converged) == "True" else " (hit max_iter=50, not converged)") for r in its.itertuples()]
    s1, a1 = section(cs, "S01_KMeans", "S01 Parallel K-means", ex1)

    q = pd.read_csv(results / "s02_knn.csv")
    acc = q.groupby(["Problem_Size", "N_Queries"]).Accuracy.agg(["min", "max"])
    ex2 = ["Accuracy (first N MNIST test images as queries; identical for every configuration and strategy):"]
    ex2 += [f"- train n={n}, {nq} queries: {r['min']:.4f}" + ("" if r["min"] == r["max"] else f" .. {r['max']:.4f} (MISMATCH)")
            for (n, nq), r in acc.iterrows()]
    qps = q[q.Parallel_Model != "sequential-chunked"].groupby(
        ["Problem_Size", "N_Queries", "Strategy", "Resource_Count"]).Queries_per_sec.mean()
    ex2.append("Peak throughput (queries/s) per train size x query count:")
    for (n, nq), g in qps.groupby(level=[0, 1]):
        i = g.idxmax()
        seqv = g.xs("none", level=2).iloc[0]
        ex2.append(f"- n={n}, {nq} q: {g.max():.0f} ({i[2]} w={i[3]}); sequential {seqv:.0f}")
    cs2 = cs[(cs.Workload == "S02_kNN") & (cs.Resource_Count == 8) & (~cs.Parallel_Model.isin(PAR_EXCLUDE))]
    ex2.append("Strategy comparison at w=8 (end-to-end speedup, query-partitioned vs train-partitioned):")
    for (ds, n), g in cs2.groupby(["Dataset", "Problem_Size"]):
        v = g.set_index("Variant").Speedup_Mean
        ex2.append(f"- {ds} n={n}: query {fmt(v.get('query-partitioned', float('nan')))}x vs "
                   f"train {fmt(v.get('train-partitioned', float('nan')))}x")
    s2, a2 = section(cs, "S02_kNN", "S02 Parallel k-NN", ex2)

    st = pd.read_csv(results / "s03_stages.csv")
    ex3 = ["Stage shares, sequential (mean wall time):"]
    seq = st[st.Parallel_Model == "sequential"].groupby(["Problem_Size", "Stage"]).Stage_Time.mean().unstack()
    for n, r in seq.iterrows():
        tot = r.sum()
        ex3.append(f"- n={n}: " + ", ".join(f"{s} {r[s]:.3f}s ({r[s] / tot:.1%})"
                                            for s in ["load", "normalise", "standardise", "hog"])
                   + f"; bottleneck = {r.idxmax()}")
    nmax = st.Problem_Size.max()
    sp = st[(st.Parallel_Model == "stage-parallel") & (st.Problem_Size == nmax)].groupby(
        ["Resource_Count", "Stage"]).Stage_Time.mean().unstack()
    ex3.append(f"Stage-parallel per-stage wall time at n={nmax} (s), vs sequential:")
    for s in ["load", "normalise", "standardise", "hog"]:
        ex3.append(f"- {s}: seq {seq.loc[nmax, s]:.3f}; " + ", ".join(f"w{w} {sp.loc[w, s]:.3f}" for w in sp.index)
                   + (f"; stage speedup@8 {seq.loc[nmax, s] / sp.loc[8, s]:.2f}x" if 8 in sp.index else ""))
    pp = st[(st.Parallel_Model == "pipeline") & (st.Problem_Size == nmax)].groupby(
        ["Resource_Count", "Stage"]).Stage_Time.mean().unstack()
    ex3.append(f"Pipeline busy CPU time per stage at n={nmax} (s, summed over replicas):")
    for w, r in pp.iterrows():
        ex3.append(f"- w{w}: " + ", ".join(f"{s} {r[s]:.3f}" for s in ["load", "normalise", "standardise", "hog"]))
    if 1 in pp.index and 8 in pp.index:
        ex3.append(f"- HOG busy CPU time inflates {pp.loc[8, 'hog'] / pp.loc[1, 'hog']:.2f}x from w1 to w8 "
                   "(same work, so per-image cost rises with 8 processes on 4P+4E cores)")
    tp = d[(d.Workload == "S03_Preprocess") & (d.Problem_Size == nmax)].assign(
        ips=lambda x: nmax / x.Execution_Time).groupby(["Variant", "Resource_Count"]).ips.mean()
    ex3.append(f"Throughput at n={nmax} (images/s): " + "; ".join(f"{m} w{w}: {v:.0f}" for (m, w), v in tp.items()))
    s3, a3 = section(cs, "S03_Preprocess", "S03 Parallel preprocessing pipeline", ex3)

    L += s1 + s2 + s3 + locality_section(cs)
    diag = results / "sampler_diagnostic.csv"
    if diag.exists():
        sd = pd.read_csv(diag)
        t = sd.groupby(["Case", "Mode"]).Time.agg(["mean", "std", "count"]).reset_index()
        L += ["## Diagnostic: memory-sampler cost on in-process (sequential) runs", "",
              "Pure sequential runs timed with the memory sampler in each mode (analysis/sampler_diagnostic.py). "
              "'scan' = original sampler (psutil children(recursive=True) every 20 ms, scans every process "
              "on the system under the parent's GIL); 'track' = only registered worker PIDs; 'off' = none.", "",
              "| Case | Mode | mean (s) | std (s) | runs | vs off |", "|---|---|---|---|---|---|"]
        for case, g in t.groupby("Case"):
            off = g[g.Mode == "off"]["mean"].iloc[0]
            for r in g.itertuples():
                L.append(f"| {case} | {r.Mode} | {r.mean:.3f} | {r.std:.3f} | {r.count} | {r.mean / off - 1:+.1%} |")
        L.append("")
    (results / "findings.md").write_text("\n".join(L) + "\n")
    print(f"wrote findings.md ({len(a1) + len(a2) + len(a3)} anomalies flagged)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    main(Path(ap.parse_args().results))
