"""Extract every number used in report tables from results/*.csv -> report/table_data.json."""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
cs = pd.read_csv(R / "config_stats.csv")
out = {}


def row(wl, ds, alg_prefix, n, var):
    g = cs[(cs.Workload == wl) & (cs.Dataset == ds) & (cs.Problem_Size == n) & (cs.Algorithm.str.startswith(alg_prefix))]
    seq = g[g.Parallel_Model == "sequential"].iloc[0]
    v = g[g.Variant == var].set_index("Resource_Count")
    return {"seq": seq.Time_Mean, "seq_std": seq.Time_Std,
            "sp": {int(w): v.loc[w, "Speedup_Mean"] for w in v.index},
            "csp": {int(w): v.loc[w, "Compute_Speedup_Mean"] for w in v.index},
            "eff8": v.loc[8, "Efficiency_Mean"], "ovh8": v.loc[8, "Overhead_Share_Mean"],
            "rss8": v.loc[8, "Memory_Use_Mean"]}


out["combined"] = [
    ["S01 K-means, MNIST, k=10", row("S01_KMeans", "MNIST", "KMeans-Lloyd-k10", 60000, "multiprocessing")],
    ["S01 K-means, Blobs-64d, k=10", row("S01_KMeans", "Blobs64d", "KMeans-Lloyd-k10", 60000, "multiprocessing")],
    ["S02 k-NN, query-partitioned", row("S02_kNN", "MNIST-q10000", "kNN", 60000, "query-partitioned")],
    ["S02 k-NN, train-partitioned", row("S02_kNN", "MNIST-q10000", "kNN", 60000, "train-partitioned")],
    ["S03 preprocessing, stage-parallel", row("S03_Preprocess", "MNIST", "load", 60000, "stage-parallel")],
    ["S03 preprocessing, pipeline", row("S03_Preprocess", "MNIST", "load", 60000, "pipeline")],
]
strat = []
for nq in (1000, 5000, 10000):
    for n in (10000, 30000, 60000):
        q = row("S02_kNN", f"MNIST-q{nq}", "kNN", n, "query-partitioned")
        t = row("S02_kNN", f"MNIST-q{nq}", "kNN", n, "train-partitioned")
        strat.append({"nq": nq, "n": n, "seq": q["seq"], "q8": q["sp"][8], "t8": t["sp"][8],
                      "qc8": q["csp"][8], "tc8": t["csp"][8], "qrss": q["rss8"], "trss": t["rss8"]})
out["strategy"] = strat
st = pd.read_csv(R / "s03_stages.csv")
st = st[st.Problem_Size == 60000]
m = st.groupby(["Parallel_Model", "Resource_Count", "Stage"]).Stage_Time.mean()
out["stages"] = {s: {"seq": m[("sequential", 1, s)], **{f"sp{w}": m[("stage-parallel", w, s)] for w in (1, 2, 4, 8)},
                     "pl1": m[("pipeline", 1, s)], "pl8": m[("pipeline", 8, s)]}
                 for s in ("load", "normalise", "standardise", "hog")}
sd = pd.read_csv(R / "sampler_diagnostic.csv").groupby(["Case", "Mode"]).Time.mean().unstack()
out["sampler"] = {c: {k: sd.loc[c, k] for k in ("off", "track", "scan")} for c in sd.index}
w1 = cs[(~cs.Parallel_Model.isin(["sequential", "sequential-chunked"])) & (cs.Resource_Count == 1)]
ch = cs[cs.Parallel_Model == "sequential-chunked"].merge(
    cs[cs.Parallel_Model == "sequential"][["Workload", "Dataset", "Problem_Size", "Base_Algorithm", "Time_Mean"]],
    on=["Workload", "Dataset", "Problem_Size", "Base_Algorithm"], suffixes=("", "_seq"))
ch["gain"] = ch.Time_Mean_seq / ch.Time_Mean - 1
out["w1_speedup_range"] = [w1.Speedup_Mean.min(), w1.Speedup_Mean.max(), int(len(w1))]
out["chunked_gain_range"] = [ch.gain.min(), ch.gain.max(), int(len(ch))]
out["cv"] = {"median": float((cs.Time_Std / cs.Time_Mean).median()), "max": float((cs.Time_Std / cs.Time_Mean).max())}
(Path(__file__).parent / "table_data.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: out[k] for k in ("w1_speedup_range", "chunked_gain_range", "cv", "sampler")}, indent=1, default=float))
