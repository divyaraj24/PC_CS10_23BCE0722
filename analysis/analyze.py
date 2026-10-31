"""Derived metrics from results/raw.csv.

Outputs:
  results/derived.csv        raw rows + Speedup, Efficiency, Compute_Speedup (per run) + baseline used
  results/config_stats.csv   mean / median / std per configuration
  results/summary_table.csv  per sub-problem x dataset x algorithm x size:
  results/summary_table.md   best time, best speedup, speedup / compute-only speedup / efficiency at 8 workers

Speedup         = mean(pure sequential Execution_Time, same Workload/Dataset/Size/base algorithm)
                  / Execution_Time
Compute_Speedup = same numerator / (Execution_Time - Scheduling_Time)
                  (removes Pool/process start-up, shutdown and dispatch latency)
Efficiency      = Speedup / Resource_Count
The base algorithm is Algorithm without a "/strategy" suffix (S02), so both k-NN
strategies share the same sequential baseline. "sequential-chunked" rows are a
diagnostic and are never used as the baseline.
"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BKEY = ["Workload", "Dataset", "Problem_Size", "Base_Algorithm"]
CFG = ["Workload", "Dataset", "Problem_Size", "Base_Algorithm", "Algorithm", "Variant",
       "Parallel_Model", "Resource_Count"]
OVH = ["Communication_Time", "Synchronization_Time", "Scheduling_Time"]


def add_keys(df):
    df = df.copy()
    df["Base_Algorithm"] = df.Algorithm.str.split("/").str[0]
    strat = df.Algorithm.str.split("/").str[1]
    df["Variant"] = strat.where(strat.notna(), df.Parallel_Model)
    return df


def derive(raw):
    raw = add_keys(raw)
    seq = (raw[raw.Parallel_Model == "sequential"].groupby(BKEY)["Execution_Time"]
           .agg(Baseline_Mean_Time="mean", Baseline_Runs="count").reset_index())
    d = raw.drop(columns=["Speedup", "Efficiency"]).merge(seq, on=BKEY, how="left")
    if d.Baseline_Mean_Time.isna().any():
        raise SystemExit("missing sequential baseline for some configuration")
    d["Speedup"] = d.Baseline_Mean_Time / d.Execution_Time
    d["Efficiency"] = d.Speedup / d.Resource_Count
    d["Compute_Speedup"] = d.Baseline_Mean_Time / (d.Execution_Time - d.Scheduling_Time)
    d["Baseline"] = ("pure sequential (no Pool), mean of " + d.Baseline_Runs.astype(str)
                     + " timed runs, same size")
    d["Overhead_Time"] = d[OVH].sum(axis=1)
    d["Overhead_Share"] = d.Overhead_Time / d.Execution_Time
    cols = [c for c in raw.columns if c not in ("Base_Algorithm", "Variant")]
    cols += ["Compute_Speedup", "Base_Algorithm", "Variant", "Baseline_Mean_Time", "Baseline",
             "Overhead_Time", "Overhead_Share"]
    return d[cols]


def stats(d):
    g = d.groupby(CFG)
    s = g["Execution_Time"].agg(Time_Mean="mean", Time_Median="median", Time_Std="std", Runs="count")
    for c in ("Speedup", "Compute_Speedup"):
        s[c + "_Mean"] = g[c].mean()
        s[c + "_Std"] = g[c].std()
    s["Efficiency_Mean"] = g["Efficiency"].mean()
    for c in OVH + ["Overhead_Share", "Memory_Use"]:
        s[c + "_Mean"] = g[c].mean()
    return s.reset_index().sort_values(CFG)


PAR_EXCLUDE = ("sequential", "sequential-chunked")


def summary(s):
    rows = []
    for key, grp in s.groupby(BKEY):
        par = grp[~grp.Parallel_Model.isin(PAR_EXCLUDE)]
        seq = grp[grp.Parallel_Model == "sequential"].iloc[0]
        bt = grp.loc[grp.Time_Mean.idxmin()]
        bs = par.loc[par.Speedup_Mean.idxmax()]
        bc = par.loc[par.Compute_Speedup_Mean.idxmax()]
        row = dict(zip(["Workload", "Dataset", "Problem_Size", "Algorithm"], key))
        row["Seq_Time_s"] = round(seq.Time_Mean, 4)
        ch = grp[grp.Parallel_Model == "sequential-chunked"]
        if len(ch):
            row["SeqChunked_Time_s"] = round(ch.Time_Mean.iloc[0], 4)
        row["Best_Time_s"] = round(bt.Time_Mean, 4)
        row["Best_Time_Config"] = f"{bt.Variant} w={bt.Resource_Count}"
        row["Best_Speedup"] = round(bs.Speedup_Mean, 3)
        row["Best_Speedup_Config"] = f"{bs.Variant} w={bs.Resource_Count}"
        row["Best_Compute_Speedup"] = round(bc.Compute_Speedup_Mean, 3)
        row["Best_Compute_Speedup_Config"] = f"{bc.Variant} w={bc.Resource_Count}"
        for var, m in par[par.Resource_Count == 8].groupby("Variant"):
            r = m.iloc[0]
            row[f"Speedup@8 {var}"] = round(r.Speedup_Mean, 3)
            row[f"ComputeSpeedup@8 {var}"] = round(r.Compute_Speedup_Mean, 3)
            row[f"Efficiency@8 {var}"] = round(r.Efficiency_Mean, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def to_md(df):
    df = df.fillna("")
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main(results):
    raw = pd.read_csv(results / "raw.csv")
    d = derive(raw)
    d.to_csv(results / "derived.csv", index=False)
    s = stats(d)
    s.to_csv(results / "config_stats.csv", index=False)
    sm = summary(s)
    sm.to_csv(results / "summary_table.csv", index=False)
    runs = int(s.Runs.mode().iloc[0])
    md = ["# Summary table (CS10, 23BCE0722)", "",
          f"Times are means of {runs} timed runs (1 warm-up discarded). Speedup baseline: pure "
          "sequential implementation (no Pool) at the same problem size. Compute speedup excludes "
          "Scheduling_Time (Pool/process start-up, shutdown, dispatch). Efficiency = speedup / workers. "
          "Variant = parallel model, or the k-NN strategy for S02.", ""]
    for wl, g in sm.groupby("Workload", sort=False):
        md += [f"## {wl}", "", to_md(g.dropna(axis=1, how="all"))]
    (results / "summary_table.md").write_text("\n".join(md))
    print(f"wrote derived.csv ({len(d)} rows), config_stats.csv ({len(s)}), summary_table.csv/.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    main(Path(ap.parse_args().results))
