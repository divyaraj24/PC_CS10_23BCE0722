"""Black-and-white architecture diagrams + console captures for the report.

Diagrams are drawn with matplotlib (PNG 300 dpi + SVG). Console captures render the
REAL text of results/run_log.txt, results/sampler_diagnostic_log.txt and the live
output of the correctness/analysis commands; nothing is typed in by hand.
    python report/make_figures.py
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none"})
K, G, W = "black", "#d9d9d9", "white"


def box(ax, x, y, w, h, text, fill=W, hatch=None, fs=7.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fill, ec=K, lw=1.0, hatch=hatch, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal",
            bbox=dict(fc="white", ec="none", pad=0.6) if hatch else None)


def arrow(ax, x1, y1, x2, y2, text=None, style="-|>", ls="-", fs=6.5, off=(0, 0.012), rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=8, color=K,
                                 lw=0.9, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], text, ha="center", va="bottom",
                fontsize=fs, style="italic")


def canvas(w=6.5, h=2.6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("  ", name)


def d_kmeans():
    fig, ax = canvas(6.5, 2.9)
    box(ax, 0.00, 0.56, 0.18, 0.26, "Parent process\ninit C$_0$: k rows\n(seed 0)", fs=7)
    box(ax, 0.00, 0.06, 0.18, 0.30, "Shared memory\nX (n x d, float32)\nlabels (n, int32)\ncopied once per run",
        fill=G, fs=6.5)
    for i, y in enumerate([0.66, 0.40, 0.04]):
        lab = f"Worker {i + 1}" if i < 2 else "Worker W"
        box(ax, 0.30, y, 0.27, 0.19, f"{lab}\nassign its block range\npartial \u03a3x, counts", fs=6.5)
        arrow(ax, 0.18, 0.69, 0.30, y + 0.095)
        arrow(ax, 0.57, y + 0.095, 0.67, 0.42)
        ax.plot([0.18, 0.30], [0.21, y + 0.095], color=K, lw=0.6, ls=":")
    ax.text(0.435, 0.315, "\u22ee", ha="center", va="center", fontsize=12)
    ax.text(0.09, 0.505, "sends C$_t$ to every worker", ha="center", fontsize=6, style="italic")
    box(ax, 0.67, 0.28, 0.17, 0.28, "Reduce (barrier)\n\u03a3 partials\nC$_{t+1}$ = \u03a3x / count", fill=G, fs=6.5)
    box(ax, 0.87, 0.28, 0.13, 0.28, "shift \u2264 tol\nor t = 50?\nyes \u2192 stop", fs=6.5)
    arrow(ax, 0.84, 0.42, 0.87, 0.42)
    ax.add_patch(FancyArrowPatch((0.755, 0.56), (0.09, 0.82), arrowstyle="-|>", mutation_scale=8, color=K,
                                 lw=0.9, connectionstyle="arc3,rad=0.25"))
    ax.text(0.50, 0.97, "no \u2192 next iteration (same persistent Pool)", ha="center", fontsize=6.5, style="italic")
    ax.text(0.09, -0.03, "dotted = read via shared memory", ha="center", fontsize=6, style="italic")
    save(fig, "diag_s01_kmeans")


def d_knn():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7))
    for ax in axs:
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.04, 1.04); ax.axis("off")
    ax = axs[0]
    ax.set_title("(a) query-partitioned", fontsize=8)
    box(ax, 0.00, 0.62, 0.27, 0.26, "Queries Q\nsplit into\nW blocks", fill=G, fs=6.5)
    box(ax, 0.00, 0.10, 0.27, 0.28, "Train X\n(all rows)\nshared mem.", fill=G, fs=6.5)
    for i, y in enumerate([0.70, 0.40, 0.10]):
        box(ax, 0.38, y, 0.30, 0.19, f"{'Worker ' + str(i + 1) if i < 2 else 'Worker W'}\nQ$_i$ vs all X\ntop-5 + vote", fs=6.3)
        arrow(ax, 0.27, 0.75, 0.38, y + 0.095)
        ax.plot([0.27, 0.38], [0.24, y + 0.095], color=K, lw=0.6, ls=":")
        arrow(ax, 0.68, y + 0.095, 0.78, 0.49)
    box(ax, 0.78, 0.36, 0.22, 0.26, "Parent:\nconcat\npredictions", fs=6.3)
    ax = axs[1]
    ax.set_title("(b) train-partitioned", fontsize=8)
    box(ax, 0.00, 0.62, 0.24, 0.26, "All Q\n(shared\nmemory)", fill=G, fs=6.5)
    box(ax, 0.00, 0.10, 0.24, 0.28, "Train X\nsplit in\nW ranges", fill=G, fs=6.5)
    for i, y in enumerate([0.70, 0.40, 0.10]):
        box(ax, 0.33, y, 0.30, 0.19, f"{'Worker ' + str(i + 1) if i < 2 else 'Worker W'}\nall Q vs X$_i$\nlocal top-5", fs=6.3)
        ax.plot([0.24, 0.33], [0.75, y + 0.095], color=K, lw=0.6, ls=":")
        arrow(ax, 0.24, 0.24, 0.33, y + 0.095)
        arrow(ax, 0.63, y + 0.095, 0.72, 0.49)
    box(ax, 0.72, 0.26, 0.28, 0.46, "Parent:\nmerge W x 5\ncandidates\n\u2192 global top-5\n\u2192 vote", fill=G, fs=6.3)
    save(fig, "diag_s02_knn")


def d_preprocess():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in axs:
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.04, 1.04); ax.axis("off")
    ax = axs[0]
    ax.set_title("(a) stage-parallel (one Pool, barrier per stage)", fontsize=8)
    stages = ["load shards", "normalise", "standardise", "HOG"]
    for r, s in enumerate(stages):
        y = 0.80 - r * 0.24
        ax.text(0.0, y + 0.06, s, fontsize=6.5, va="center")
        for c in range(4):
            box(ax, 0.30 + c * 0.17, y, 0.14, 0.12, "w" + (str(c + 1) if c < 3 else "W"), fill=G if r == 3 else W, fs=6)
        if r < 3:
            ax.plot([0.28, 0.99], [y - 0.06, y - 0.06], color=K, lw=1.2, ls="--")
    ax.text(0.99, 0.02, "-- -- barrier (pool.map returns)", fontsize=6, ha="right", style="italic")
    ax = axs[1]
    ax.set_title("(b) pipeline (processes + Queues, batches of 1000)", fontsize=8)
    xs = [0.00, 0.19, 0.38]
    for x, s in zip(xs, ["Load", "Norm-\nalise", "Stand-\nardise"]):
        box(ax, x, 0.40, 0.15, 0.22, s + "\n1 proc", fs=6.2)
    for x0, x1 in [(0.15, 0.19), (0.34, 0.38)]:
        arrow(ax, x0, 0.51, x1, 0.51)
    for i, y in enumerate([0.74, 0.42, 0.10]):
        box(ax, 0.60, y, 0.15, 0.16, "HOG " + (str(i + 1) if i < 2 else "W"), fill=G, fs=6.5)
        arrow(ax, 0.53, 0.51, 0.60, y + 0.08)
        arrow(ax, 0.75, y + 0.08, 0.82, 0.51)
    box(ax, 0.82, 0.38, 0.18, 0.26, "Parent\ncollect,\nre-order", fs=6.2)
    ax.text(0.40, 0.70, "Queues, maxsize 8", fontsize=6, ha="center", style="italic")
    ax.text(0.5, 0.0, "processes = W HOG replicas + 3 front stages", fontsize=6, ha="center", style="italic")
    save(fig, "diag_s03_preprocess")


def d_protocol():
    fig, ax = canvas(6.5, 2.2)
    steps = ["Config:\nsize, model,\nworkers", "Warm-up run\n(checked,\ndiscarded)",
             "5 timed runs\n+ peak-RSS\nsampler", "Correctness\ncheck vs\nsequential",
             "raw.csv\n(19 columns)", "analyze.py:\nspeedup,\nefficiency"]
    w, gap = 0.145, 0.026
    for i, s in enumerate(steps):
        x = i * (w + gap)
        box(ax, x, 0.55, w, 0.36, s, fill=G if i in (3,) else W, fs=6.3)
        if i:
            arrow(ax, x - gap, 0.73, x, 0.73)
    ax.text(0.0, 0.40, "Execution_Time per run =", fontsize=7, fontweight="bold")
    parts = [("Computation", W, None), ("Communication\nshm copies, results", G, None),
             ("Synchronization\nbarrier idle", W, "////"), ("Scheduling\nPool start/stop,\ndispatch", G, "...")]
    x = 0.0
    for (t, f, h), wd in zip(parts, [0.30, 0.25, 0.20, 0.25]):
        box(ax, x, 0.05, wd - 0.01, 0.28, t, fill=f, hatch=h, fs=6.3)
        x += wd
    save(fig, "diag_protocol")


def capture(name, title, blocks, width=6.5, fs=6.2):
    """blocks: list of (prompt_line, output_lines)."""
    lines = []
    for cmd, out in blocks:
        lines.append(("$ " + cmd, True))
        lines += [(l.rstrip("\n"), False) for l in out]
    h = 0.35 + 0.118 * len(lines) * fs / 6.2
    fig = plt.figure(figsize=(width, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.003, 0.003), 0.994, 0.994, fill=False, ec=K, lw=1.2, transform=ax.transAxes))
    bar = 0.22 / h
    ax.add_patch(plt.Rectangle((0.003, 1 - bar), 0.994, bar - 0.003, fc=G, ec=K, lw=1.2, transform=ax.transAxes))
    ax.text(0.5, 1 - bar / 2, title, ha="center", va="center", fontsize=7, transform=ax.transAxes)
    y = 1 - bar - 0.06 / h
    step = (1 - bar - 0.1 / h) / max(len(lines), 1)
    for t, is_cmd in lines:
        ax.text(0.015, y, t, family="DejaVu Sans Mono", fontsize=fs, va="top", transform=ax.transAxes,
                parse_math=False,
                fontweight="bold" if is_cmd else "normal")
        y -= step
    save(fig, name)


def captures():
    res = ROOT / "results"
    log = (res / "run_log.txt").read_text().splitlines()
    i0 = next(i for i, l in enumerate(log) if l.startswith("[S02] train=60000 queries=10000"))
    seq = [l for l in log[i0:] if "q10000-60000-kNN-brute-k5-sequential-w1-r" in l][:5]
    tp8 = [l for l in log[i0:] if "q10000-60000-kNN-brute-k5/train-partitioned-multiprocessing-w8-r" in l][:5]
    short = lambda l: l.replace("S02_kNN-MNIST-q10000-60000-kNN-brute-k5", "S02 q10000 n60000 ").replace("    ", "  ")
    capture("cap_sweep", "zsh - cs10: final benchmark sweep (excerpt of results/run_log.txt)", [
        ("python run_all.py --mem-sampler track", [log[0].split("->")[0] + "-> results/raw.csv", "..."]
         + [log[i0]] + [short(l) for l in seq] + ["  ..."] + [short(l) for l in tp8] + ["  ..."] + log[-3:]),
    ])
    corr = subprocess.run(["awk", "-F,", 'NR>1{c[$9]++} END{for(k in c) print "Passed="k, c[k]}',
                           str(res / "correctness.csv")], capture_output=True, text=True).stdout.splitlines()
    rows = subprocess.run(["wc", "-l", str(res / "raw.csv")], capture_output=True, text=True).stdout.split()[0]
    py = str(ROOT / ".venv" / "bin" / "python")
    ana = subprocess.run([py, str(ROOT / "analysis" / "analyze.py")], capture_output=True, text=True,
                         cwd=ROOT).stdout.splitlines()
    fnd = subprocess.run([py, str(ROOT / "analysis" / "findings.py")], capture_output=True, text=True,
                         cwd=ROOT).stdout.splitlines()
    capture("cap_correctness", "zsh - cs10: correctness log and analysis", [
        ("awk -F, 'NR>1{c[$9]++} END{for(k in c) print \"Passed=\"k, c[k]}' results/correctness.csv", corr),
        ("wc -l results/raw.csv", [f"{rows} results/raw.csv   (header + {int(rows) - 1} timed runs)"]),
        ("python analysis/analyze.py", ana),
        ("python analysis/findings.py", fnd),
    ])
    diag = (res / "sampler_diagnostic_log.txt").read_text().splitlines()
    capture("cap_sampler", "zsh - cs10: memory-sampler diagnostic (results/sampler_diagnostic_log.txt)", [
        ("python analysis/sampler_diagnostic.py --size 60000 --repeats 3", diag),
    ], fs=5.8)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d_kmeans(); d_knn(); d_preprocess(); d_protocol(); captures()
