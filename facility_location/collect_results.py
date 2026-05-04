"""
collect_results.py
==================
Reads all comparison_summary_G_*.csv files from results/<TYPE>/ and
produces two formatted tables:

  1. Out-of-sample performance  (Avg_Profit ± Std_Dev, gap DDU vs STD)
  2. Algorithm diagnostics      (Final_LB, Iterations, Wall_Time_s)

Usage
-----
    python collect_results.py              # pretty-print to console
    python collect_results.py --latex      # also write results/summary_table.tex
    python collect_results.py --csv        # also write results/summary_table.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd

# ── Locate result files ────────────────────────────────────────────────────────
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results", "small_D")
pattern = os.path.join(RESULTS_ROOT, "**", "comparison_summary_*.csv")
files = sorted(glob.glob(pattern, recursive=True))

if not files:
    sys.exit(
        f"No result files found under {RESULTS_ROOT}.\n"
        "Run run_all.sh (or individual julia calls) first."
    )

print(f"Found {len(files)} result file(s):")
for f in files:
    print(f"  {os.path.relpath(f, RESULTS_ROOT)}")
print()

# ── Load and concatenate ───────────────────────────────────────────────────────
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Normalise: exact-scenario runs use "Exact_Scenarios" instead of "N_SAA"
if "Exact_Scenarios" in df.columns and "N_SAA" not in df.columns:
    df = df.rename(columns={"Exact_Scenarios": "N_SAA"})
elif "Exact_Scenarios" in df.columns:
    df["N_SAA"] = df["N_SAA"].fillna(df["Exact_Scenarios"])
    df = df.drop(columns=["Exact_Scenarios"])

# Ensure consistent ordering
df = df.sort_values(["Instance_Type", "N_SAA", "Policy"]).reset_index(drop=True)

# ── Table 1: Out-of-sample performance ────────────────────────────────────────
perf = df.pivot_table(
    index=["Instance_Type", "N_SAA"],
    columns="Policy",
    values=["Avg_Profit", "Std_Dev"],
    aggfunc="first",
)

# Flatten MultiIndex columns
perf.columns = [f"{stat}_{pol}" for stat, pol in perf.columns]
perf = perf.reset_index()

perf["Gap_%"] = (
    (perf["Avg_Profit_DDU-SDDiP"] - perf["Avg_Profit_Standard SDDiP"])
    / perf["Avg_Profit_Standard SDDiP"].abs()
    * 100
).round(2)

perf_display = perf.rename(columns={
    "Instance_Type":              "Type",
    "Avg_Profit_DDU-SDDiP":      "DDU Avg Profit",
    "Std_Dev_DDU-SDDiP":         "DDU Std",
    "Avg_Profit_Standard SDDiP": "STD Avg Profit",
    "Std_Dev_Standard SDDiP":    "STD Std",
    "Gap_%":                     "Gap (%)",
})
for col in ["DDU Avg Profit", "DDU Std", "STD Avg Profit", "STD Std"]:
    perf_display[col] = perf_display[col].round(1)

print("=" * 72)
print("TABLE 1 — Out-of-sample performance (1 000-path simulation)")
print("=" * 72)
print(perf_display.to_string(index=False))
print()

# ── Table 2: Algorithm diagnostics ────────────────────────────────────────────
diag = df.pivot_table(
    index=["Instance_Type", "N_SAA"],
    columns="Policy",
    values=["Final_LB", "Iterations", "Wall_Time_s"],
    aggfunc="first",
)
diag.columns = [f"{stat}_{pol}" for stat, pol in diag.columns]
diag = diag.reset_index()

diag_display = diag.rename(columns={
    "Instance_Type":              "Type",
    "Final_LB_DDU-SDDiP":        "DDU LB",
    "Final_LB_Standard SDDiP":   "STD LB",
    "Iterations_DDU-SDDiP":      "DDU Iters",
    "Iterations_Standard SDDiP": "STD Iters",
    "Wall_Time_s_DDU-SDDiP":     "DDU Time (s)",
    "Wall_Time_s_Standard SDDiP":"STD Time (s)",
})
for col in ["DDU LB", "STD LB"]:
    diag_display[col] = diag_display[col].round(1)

print("=" * 72)
print("TABLE 2 — Algorithm diagnostics")
print("=" * 72)
print(diag_display.to_string(index=False))
print()

# ── Table 3: First-stage decisions ────────────────────────────────────────────
dec = df[["Instance_Type", "N_SAA", "Policy", "First_Stage_Facs", "Active_Region"]].copy()
dec_display = dec.rename(columns={
    "Instance_Type": "Type",
    "First_Stage_Facs": "Facilities (1-indexed)",
    "Active_Region": "Region",
})

print("=" * 72)
print("TABLE 3 — First-stage facility decisions")
print("=" * 72)
print(dec_display.to_string(index=False))
print()

# ── Optional outputs ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--latex", action="store_true", help="Write LaTeX tables")
parser.add_argument("--csv",   action="store_true", help="Write combined CSV")
args = parser.parse_args()

os.makedirs(RESULTS_ROOT, exist_ok=True)

if args.csv:
    out_csv = os.path.join(RESULTS_ROOT, "summary_all.csv")
    df.to_csv(out_csv, index=False)
    print(f"Combined CSV written to: {out_csv}")

if args.latex:
    tex_path = os.path.join(RESULTS_ROOT, "summary_tables.tex")
    with open(tex_path, "w") as fh:
        fh.write("% Table 1 — Out-of-sample performance\n")
        fh.write(perf_display.to_latex(index=False, float_format="%.1f"))
        fh.write("\n\n% Table 2 — Algorithm diagnostics\n")
        fh.write(diag_display.to_latex(index=False, float_format="%.1f"))
        fh.write("\n\n% Table 3 — First-stage decisions\n")
        fh.write(dec_display.to_latex(index=False))
    print(f"LaTeX tables written to: {tex_path}")
