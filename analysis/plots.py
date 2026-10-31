"""All graphs (PNG 300 dpi + SVG) from results/derived.csv and side CSVs.

S01: time vs samples, speedup vs workers, iterations vs configuration
S02: time vs dataset size, throughput vs workers, speedup vs workers
S03: stage contribution (stacked), speedup vs workers, throughput per variant
+ overhead breakdown (computation vs communication/sync/scheduling) at the largest size.
Error bars = std over the 3 timed runs.
"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W_FULL, H = 6.5, 3.6  # inches; fits a Word page with 1" margins
WORKERS = [1, 2, 4, 8]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "lines.markersize": 5, "legend.frameon": False, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


PAR_EXCLUDE = ("sequential", "sequential-chunked")


def agg(d, cols=("Execution_Time", "Speedup", "Compute_Speedup")):
    keys = ["Workload", "Dataset", "Problem_Size", "Algorithm", "Variant", "Parallel_Model", "Resource_Count"]
    g = d.groupby(keys)[list(cols)].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def series_label(model, w):
    return "sequential" if model == "sequential" else f"{model} w={w}"


def time_vs_size(ax, a, title):
    """Lines: sequential + one per worker count; x = problem size."""
    configs = [("sequential", 1)] + [(m, w) for m in a.Variant.unique() if m not in PAR_EXCLUDE
                                     for w in WORKERS]
    for i, (m, w) in enumerate(configs):
        s = a[(a.Variant == m) & (a.Resource_Count == w)].sort_values("Problem_Size")
        if s.empty:
            continue
        ax.errorbar(s.Problem_Size, s.Execution_Time_mean, yerr=s.Execution_Time_std.fillna(0),
                    color=PALETTE[i], marker=MARKERS[i], capsize=2,
                    linestyle="--" if m == "sequential" else "-", label=series_label(m, w))
    ax.set_title(title)
    ax.set_xlabel("Problem size (samples)")
    ax.set_ylabel("Execution time (s)")
    ax.set_xticks(sorted(a.Problem_Size.unique()))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))


def speedup_vs_workers(ax, a, group_col, title, label_fmt="{}"):
    for i, (gv, s) in enumerate(a[~a.Parallel_Model.isin(PAR_EXCLUDE)].groupby(group_col, sort=True)):
        s = s.sort_values("Resource_Count")
        ax.errorbar(s.Resource_Count, s.Speedup_mean, yerr=s.Speedup_std.fillna(0), color=PALETTE[i],
                    marker=MARKERS[i], capsize=2, label=label_fmt.format(gv))
    ax.plot(WORKERS, WORKERS, color=INK2, linestyle=":", linewidth=1, label="ideal (linear)")
    ax.axhline(1.0, color=INK2, linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(WORKERS)
    ax.set_xticklabels(WORKERS)
    ax.set_xlabel("Workers (processes)")
    ax.set_ylabel("Speedup vs sequential")
    ax.set_title(title)


def size_fmt(n):
    return f"{int(n) // 1000}k samples"


def main(results, out):
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(results / "derived.csv")
    a = agg(d)

    # ---------------- S01 ----------------
    s1 = a[(a.Workload == "S01_KMeans") & (a.Algorithm == "KMeans-Lloyd-k10")]
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, H))
    for ax, ds in zip(axs, ["MNIST", "Blobs64d"]):
        time_vs_size(ax, s1[s1.Dataset == ds], f"K-means (k=10), {ds}")
    axs[1].legend(loc="upper left")
    save(fig, out, "s01_time_vs_samples")

    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, H), sharey=True)
    for ax, ds in zip(axs, ["MNIST", "Blobs64d"]):
        s = s1[s1.Dataset == ds].copy()
        s["Size"] = s.Problem_Size.map(size_fmt)
        speedup_vs_workers(ax, s.sort_values("Problem_Size"), "Problem_Size", f"K-means (k=10), {ds}",
                           "n={}")
    axs[0].legend(loc="upper left")
    save(fig, out, "s01_speedup_vs_workers")

    it = pd.read_csv(results / "s01_kmeans.csv")
    it = it[it.Parallel_Model == "sequential"].groupby(["Dataset", "Problem_Size", "k"]).Iterations.first().reset_index()
    it = it.sort_values(["Dataset", "Problem_Size", "k"])
    labels = [f"{r.Dataset}\nn={r.Problem_Size // 1000}k, k={r.k}" for r in it.itertuples()]
    fig, ax = plt.subplots(figsize=(W_FULL, H))
    bars = ax.bar(range(len(it)), it.Iterations, color=PALETTE[0], width=0.6)
    ax.bar_label(bars, padding=2, fontsize=8, color=INK)
    ax.axhline(50, color=INK2, linestyle=":", linewidth=1, label="max_iter = 50 (cap)")
    ax.legend(loc="upper left")
    ax.set_xticks(range(len(it)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Iterations to convergence (tol=1e-4)")
    ax.set_title("K-means iterations per configuration (identical for sequential and all worker counts)")
    ax.set_ylim(0, 62)
    save(fig, out, "s01_iterations_vs_config")

    # ---------------- S02 ----------------
    s2 = a[a.Workload == "S02_kNN"].copy()
    s2["NQ"] = s2.Dataset.str.extract(r"q(\d+)")[0].astype(int)
    qs = sorted(s2.NQ.unique())
    qmax, nmax2 = max(qs), s2.Problem_Size.max()
    STRATS = ["query-partitioned", "train-partitioned"]

    fig, axs = plt.subplots(1, len(qs), figsize=(W_FULL, H), sharey=False)
    for ax, nq in zip(np.atleast_1d(axs), qs):
        time_vs_size(ax, s2[(s2.NQ == nq) & s2.Variant.isin(["sequential", "query-partitioned"])],
                     f"{nq} queries")
        ax.set_xlabel("Train size")
        if nq != qs[0]:
            ax.set_ylabel("")
    np.atleast_1d(axs)[0].legend(loc="upper left", fontsize=6)
    fig.suptitle("k-NN (k=5) execution time, query-partitioned", fontsize=10)
    save(fig, out, "s02_time_vs_size")

    xs = [0] + WORKERS
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, H), sharey=True)
    for ax, strat in zip(axs, STRATS):
        for i, (n, g) in enumerate(s2[s2.NQ == qmax].groupby("Problem_Size")):
            seq = g[g.Variant == "sequential"]
            par = g[g.Variant == strat].set_index("Resource_Count").reindex(WORKERS)
            t_mean = [seq.Execution_Time_mean.iloc[0]] + list(par.Execution_Time_mean)
            t_std = [seq.Execution_Time_std.iloc[0]] + list(par.Execution_Time_std)
            tp = qmax / np.array(t_mean)
            err = tp * np.array(t_std) / np.array(t_mean)  # first-order std of q/t
            ax.errorbar(range(len(xs)), tp, yerr=np.nan_to_num(err), color=PALETTE[i], marker=MARKERS[i],
                        capsize=2, label=f"train n={n // 1000}k")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(["seq"] + [str(w) for w in WORKERS])
        ax.set_xlabel("Workers (seq = sequential, no Pool)")
        ax.set_title(f"{strat}, {qmax} queries")
    axs[0].set_ylabel("Throughput (queries / s)")
    axs[0].legend(loc="upper left")
    save(fig, out, "s02_throughput_vs_workers")

    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, H), sharey=True)
    for ax, strat in zip(axs, STRATS):
        sub = s2[(s2.NQ == qmax) & (s2.Variant == strat)].sort_values("Problem_Size")
        speedup_vs_workers(ax, sub, "Problem_Size", f"{strat}, {qmax} queries", "train n={}")
    axs[1].set_ylabel("")
    axs[0].legend(loc="upper left")
    save(fig, out, "s02_speedup_vs_workers")

    fig, ax = plt.subplots(figsize=(W_FULL, H))
    sub = s2[(s2.NQ == qmax) & (s2.Problem_Size == nmax2)]
    speedup_vs_workers(ax, sub, "Variant",
                       f"k-NN strategy comparison: train n={nmax2 // 1000}k, {qmax} queries")
    ax.legend(loc="upper left")
    save(fig, out, "s02_strategy_comparison")

    # ---------------- S03 ----------------
    st = pd.read_csv(results / "s03_stages.csv")
    stages = ["load", "normalise", "standardise", "hog"]
    nmax = st.Problem_Size.max()
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, H), gridspec_kw={"width_ratios": [3, 4]})
    left = st[st.Parallel_Model == "sequential"].groupby(["Problem_Size", "Stage"]).Stage_Time.mean().unstack()[stages]
    right = st[(st.Parallel_Model == "stage-parallel") & (st.Problem_Size == nmax)] \
        .groupby(["Resource_Count", "Stage"]).Stage_Time.mean().unstack()[stages]
    for ax, tab, xl, title in [(axs[0], left, [f"{n // 1000}k" for n in left.index], "Sequential, by size"),
                               (axs[1], right, [f"w={w}" for w in right.index], f"Stage-parallel, {nmax // 1000}k images")]:
        bottom = np.zeros(len(tab))
        for i, sname in enumerate(stages):
            ax.bar(range(len(tab)), tab[sname], bottom=bottom, color=PALETTE[i], width=0.6,
                   edgecolor="white", linewidth=1, label=sname)
            bottom += tab[sname].values
        for x, tot in enumerate(bottom):
            hog_share = tab["hog"].iloc[x] / tot
            ax.text(x, tot, f"{tot:.2f}s\nHOG {hog_share:.0%}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(tab)))
        ax.set_xticklabels(xl)
        ax.set_title(title)
        ax.set_ylabel("Stage wall time (s)")
        ax.set_ylim(0, bottom.max() * 1.25)
    axs[0].legend(loc="upper left")
    save(fig, out, "s03_stage_contribution")

    s3 = a[a.Workload == "S03_Preprocess"]
    sizes = sorted(s3.Problem_Size.unique())
    fig, axs = plt.subplots(1, len(sizes), figsize=(W_FULL, H), sharey=True)
    for ax, n in zip(np.atleast_1d(axs), sizes):
        speedup_vs_workers(ax, s3[s3.Problem_Size == n], "Variant", f"{n // 1000}k images")
        if ax is not np.atleast_1d(axs)[0]:
            ax.set_ylabel("")
    np.atleast_1d(axs)[0].legend(loc="upper left")
    save(fig, out, "s03_speedup_vs_workers")

    d3 = d[(d.Workload == "S03_Preprocess") & (d.Problem_Size == nmax)].copy()
    d3["Images_per_sec"] = nmax / d3.Execution_Time
    t3 = d3.groupby(["Parallel_Model", "Resource_Count"]).Images_per_sec.agg(["mean", "std"]).reset_index()
    seq_tp = t3[t3.Parallel_Model == "sequential"]["mean"].iloc[0]
    fig, ax = plt.subplots(figsize=(W_FULL, H))
    width = 0.36
    for i, model in enumerate(["stage-parallel", "pipeline"]):
        s = t3[t3.Parallel_Model == model].set_index("Resource_Count").reindex(WORKERS)
        x = np.arange(len(WORKERS)) + (i - 0.5) * (width + 0.02)
        b = ax.bar(x, s["mean"], width, yerr=s["std"].fillna(0), color=PALETTE[i], capsize=2, label=model)
        ax.bar_label(b, fmt="%.0f", fontsize=7, padding=2)
    ax.axhline(seq_tp, color=INK2, linestyle="--", linewidth=1, label=f"sequential ({seq_tp:.0f}/s)")
    ax.set_xticks(range(len(WORKERS)))
    ax.set_xticklabels([f"w={w}" for w in WORKERS])
    ax.set_ylabel("Throughput (images / s)")
    ax.set_xlabel("Workers (pipeline: w HOG replicas + 3 single-process front stages)")
    ax.set_title(f"Preprocessing throughput per variant, {nmax // 1000}k images")
    ax.legend(loc="upper left")
    save(fig, out, "s03_throughput_per_variant")

    # ---------------- overhead breakdown (all three) ----------------
    ov = d.groupby(["Workload", "Dataset", "Problem_Size", "Algorithm", "Variant", "Resource_Count"])[
        ["Execution_Time", "Communication_Time", "Synchronization_Time", "Scheduling_Time"]].mean().reset_index()
    ov["Computation"] = (ov.Execution_Time - ov[["Communication_Time", "Synchronization_Time",
                                                  "Scheduling_Time"]].sum(axis=1)).clip(lower=0)
    q2 = f"MNIST-q{qmax}"
    panels = [("S01_KMeans", "MNIST", "KMeans-Lloyd-k10", "multiprocessing", "S01 K-means\nMNIST k=10"),
              ("S02_kNN", q2, None, "query-partitioned", f"S02 k-NN query-\npart., {qmax} q"),
              ("S02_kNN", q2, None, "train-partitioned", f"S02 k-NN train-\npart., {qmax} q"),
              ("S03_Preprocess", "MNIST", None, "stage-parallel", "S03 stage-\nparallel"),
              ("S03_Preprocess", "MNIST", None, "pipeline", "S03\npipeline")]
    parts = [("Computation", PALETTE[0]), ("Communication_Time", PALETTE[1]),
             ("Synchronization_Time", PALETTE[2]), ("Scheduling_Time", PALETTE[3])]
    fig, axs = plt.subplots(1, len(panels), figsize=(W_FULL, H + 0.4))
    for ax, (wl, ds, alg, var, title) in zip(axs, panels):
        s = ov[(ov.Workload == wl) & (ov.Dataset == ds)]
        if alg:
            s = s[s.Algorithm == alg]
        n = s.Problem_Size.max()
        s = s[(s.Problem_Size == n) & (s.Variant == var)].set_index("Resource_Count").reindex(WORKERS)
        bottom = np.zeros(len(s))
        for col, c in parts:
            ax.bar(range(len(s)), s[col], bottom=bottom, color=c, width=0.65, edgecolor="white",
                   linewidth=1, label=col.replace("_Time", ""))
            bottom += s[col].values
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(WORKERS)
        ax.set_xlabel("Workers")
        ax.set_title(f"{title}\n(n={n // 1000}k)", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
    axs[0].set_ylabel("Mean time (s)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out / f"overhead_breakdown.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("   overhead_breakdown")

    # ---------------- end-to-end vs compute-only speedup ----------------
    groups = [("S01 K-means, MNIST k=10", a[(a.Workload == "S01_KMeans") & (a.Dataset == "MNIST")
                                           & (a.Algorithm == "KMeans-Lloyd-k10")], ["multiprocessing"]),
              (f"S02 k-NN, {qmax} queries", s2[s2.NQ == qmax], STRATS),
              ("S03 preprocessing", a[a.Workload == "S03_Preprocess"], ["stage-parallel", "pipeline"])]
    fig, axs = plt.subplots(1, 3, figsize=(W_FULL, H), sharey=True)
    for ax, (title, g, variants) in zip(axs, groups):
        n = g.Problem_Size.max()
        g = g[g.Problem_Size == n]
        for i, var in enumerate(variants):
            v = g[g.Variant == var].sort_values("Resource_Count")
            ax.errorbar(v.Resource_Count, v.Speedup_mean, yerr=v.Speedup_std.fillna(0), color=PALETTE[i],
                        marker=MARKERS[i], capsize=2, label=f"{var} end-to-end")
            ax.errorbar(v.Resource_Count, v.Compute_Speedup_mean, yerr=v.Compute_Speedup_std.fillna(0),
                        color=PALETTE[i], marker=MARKERS[i], markerfacecolor="white", linestyle="--",
                        capsize=2, label=f"{var} compute-only")
        ax.plot(WORKERS, WORKERS, color=INK2, linestyle=":", linewidth=1, label="ideal")
        ax.axhline(1.0, color=INK2, linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(WORKERS)
        ax.set_xticklabels(WORKERS)
        ax.set_xlabel("Workers")
        ax.set_title(f"{title}\n(n={n // 1000}k)", fontsize=8)
        ax.legend(loc="upper left", fontsize=5.5)
    axs[0].set_ylabel("Speedup vs sequential")
    save(fig, out, "speedup_e2e_vs_compute")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "plots"))
    a = ap.parse_args()
    main(Path(a.results), Path(a.out))
