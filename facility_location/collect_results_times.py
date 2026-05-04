"""
collect_results_times.py
========================
Reads results produced by run_comparison_times.jl and prints formatted tables:

  1. Out-of-sample performance   (Avg_Profit ± Std_Dev, gap DDU vs STD)
  2. Algorithm diagnostics        (Final_LB, Iterations, Wall_Time_s)
  3. Per-pass timing summary      (mean / median / p95 / total for fwd and bwd)
  4. Per-stage opening summary    (avg # open facilities, modal pattern)
  5. Per-facility open frequency  (full table, one row per policy/stage/facility)

Usage
-----
    python collect_results_times.py              # pretty-print to console
    python collect_results_times.py --latex      # also write results/summary_times.tex
    python collect_results_times.py --csv        # also write results/summary_times_all.csv
    python collect_results_times.py --plot       # also write timing / opening PNG charts
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "small_D")
FILE_TAG    = "SMALL_D_exact_times"


def _path(name):
    return os.path.join(RESULTS_DIR, f"{name}_{FILE_TAG}.csv")


def _require(path):
    if not os.path.exists(path):
        sys.exit(
            f"Required file not found: {path}\n"
            "Run run_comparison_times.jl first."
        )
    return path


summary_path  = _require(_path("comparison_summary"))
conv_path     = _require(_path("convergence_data"))
open_stats_path  = _require(_path("opening_stats"))
open_summ_path   = _require(_path("opening_summary"))

# ── Load data ─────────────────────────────────────────────────────────────────
df_summary   = pd.read_csv(summary_path)
df_conv      = pd.read_csv(conv_path)
df_open_stat = pd.read_csv(open_stats_path)
df_open_summ = pd.read_csv(open_summ_path)

# Normalise column name used in other scripts
if "Exact_Scenarios" in df_summary.columns and "N_SAA" not in df_summary.columns:
    df_summary = df_summary.rename(columns={"Exact_Scenarios": "N_SAA"})

# ── Table 1: Out-of-sample performance ───────────────────────────────────────
perf = df_summary.pivot_table(
    index=["Instance_Type", "N_SAA"],
    columns="Policy",
    values=["Avg_Profit", "Std_Dev"],
    aggfunc="first",
)
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

# ── Table 2: Algorithm diagnostics ───────────────────────────────────────────
diag = df_summary.pivot_table(
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

# ── Table 3: Per-pass timing summary ─────────────────────────────────────────
# Summarise fwd_time_s and bwd_time_s per policy across all iterations.

def timing_row(grp, policy):
    fwd = grp["fwd_time_s"].dropna()
    bwd = grp["bwd_time_s"].dropna()
    return {
        "Policy":          policy,
        "N_iters":         len(fwd),
        # Forward pass
        "Fwd_mean_s":      round(fwd.mean(),               4),
        "Fwd_median_s":    round(fwd.median(),             4),
        "Fwd_p95_s":       round(fwd.quantile(0.95),       4),
        "Fwd_max_s":       round(fwd.max(),                4),
        "Fwd_total_s":     round(fwd.sum(),                2),
        # Backward pass
        "Bwd_mean_s":      round(bwd.mean(),               4),
        "Bwd_median_s":    round(bwd.median(),             4),
        "Bwd_p95_s":       round(bwd.quantile(0.95),       4),
        "Bwd_max_s":       round(bwd.max(),                4),
        "Bwd_total_s":     round(bwd.sum(),                2),
        # Ratio
        "Bwd_Fwd_ratio":   round(bwd.mean() / fwd.mean(), 2) if fwd.mean() > 0 else float("nan"),
    }

timing_rows = []
for policy, grp in df_conv.groupby("policy"):
    timing_rows.append(timing_row(grp, policy))
df_timing = pd.DataFrame(timing_rows)

# Wide format display — split into two sub-tables for readability
fwd_cols = ["Policy", "N_iters", "Fwd_mean_s", "Fwd_median_s", "Fwd_p95_s",
            "Fwd_max_s", "Fwd_total_s"]
bwd_cols = ["Policy", "Bwd_mean_s", "Bwd_median_s", "Bwd_p95_s",
            "Bwd_max_s", "Bwd_total_s", "Bwd_Fwd_ratio"]

print("=" * 72)
print("TABLE 3a — Forward-pass timing (seconds per iteration)")
print("=" * 72)
print(df_timing[fwd_cols].to_string(index=False))
print()
print("=" * 72)
print("TABLE 3b — Backward-pass timing (seconds per iteration)")
print("=" * 72)
print(df_timing[bwd_cols].to_string(index=False))
print()

# ── Table 4: Per-stage opening summary ───────────────────────────────────────
print("=" * 72)
print("TABLE 4 — Per-stage opening summary (out-of-sample simulation)")
print("=" * 72)
print(df_open_summ.to_string(index=False))
print()

# ── Table 5: Per-facility open frequency ─────────────────────────────────────
# Pivot to (Stage × Facility) with one column per policy.
freq_pivot = df_open_stat.pivot_table(
    index=["Stage", "Facility"],
    columns="Policy",
    values="Open_Frequency",
    aggfunc="first",
).reset_index()
freq_pivot.columns.name = None
freq_pivot = freq_pivot.round(3)

# Sort facilities in natural order within each stage
freq_pivot = freq_pivot.sort_values(["Stage", "Facility"]).reset_index(drop=True)

print("=" * 72)
print("TABLE 5 — Per-facility open frequency at each stage (simulation)")
print("=" * 72)
print(freq_pivot.to_string(index=False))
print()

# ── Optional outputs ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--latex", action="store_true", help="Write LaTeX tables")
parser.add_argument("--csv",   action="store_true", help="Write combined CSV")
parser.add_argument("--plot",  action="store_true", help="Write timing and opening plots")
args = parser.parse_args()

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── CSV output ────────────────────────────────────────────────────────────────
if args.csv:
    all_csv = os.path.join(RESULTS_DIR, f"summary_times_all_{FILE_TAG}.csv")
    combined = df_summary.copy()
    combined.to_csv(all_csv, index=False)
    print(f"Combined summary CSV written to: {all_csv}")

    timing_csv = os.path.join(RESULTS_DIR, f"timing_summary_{FILE_TAG}.csv")
    df_timing.to_csv(timing_csv, index=False)
    print(f"Timing summary CSV written to: {timing_csv}")

# ── LaTeX output ─────────────────────────────────────────────────────────────
if args.latex:
    tex_path = os.path.join(RESULTS_DIR, f"summary_times_{FILE_TAG}.tex")
    with open(tex_path, "w") as fh:
        fh.write("% Table 1 — Out-of-sample performance\n")
        fh.write(perf_display.to_latex(index=False, float_format="%.1f"))
        fh.write("\n\n% Table 2 — Algorithm diagnostics\n")
        fh.write(diag_display.to_latex(index=False, float_format="%.1f"))
        fh.write("\n\n% Table 3a — Forward-pass timing\n")
        fh.write(df_timing[fwd_cols].to_latex(index=False, float_format="%.4f"))
        fh.write("\n\n% Table 3b — Backward-pass timing\n")
        fh.write(df_timing[bwd_cols].to_latex(index=False, float_format="%.4f"))
        fh.write("\n\n% Table 4 — Per-stage opening summary\n")
        fh.write(df_open_summ.to_latex(index=False))
        fh.write("\n\n% Table 5 — Per-facility open frequency\n")
        fh.write(freq_pivot.to_latex(index=False, float_format="%.3f"))
    print(f"LaTeX tables written to: {tex_path}")

# ── Plot output ───────────────────────────────────────────────────────────────
if args.plot:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        sys.exit("matplotlib is required for --plot. Install with: pip install matplotlib")

    POLICIES   = df_conv["policy"].unique().tolist()
    COLORS     = {"DDU-SDDiP": "#1f77b4", "Standard SDDiP": "#d62728"}
    LINESTYLES = {"DDU-SDDiP": "-",        "Standard SDDiP": "--"}

    # ── Plot 1: LB convergence (iteration and time) ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for pol in POLICIES:
        grp = df_conv[df_conv["policy"] == pol]
        c   = COLORS.get(pol, "grey")
        ls  = LINESTYLES.get(pol, "-")
        axes[0].plot(grp["iteration"],  grp["lower_bound"],
                     label=pol, color=c, linestyle=ls, linewidth=1.8)
        axes[1].plot(grp["time_s"],     grp["lower_bound"],
                     label=pol, color=c, linestyle=ls, linewidth=1.8)
    for ax, xlabel in zip(axes, ["Iteration", "Wall time (s)"]):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Lower Bound")
        ax.set_title(f"{xlabel} vs Lower Bound")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    p1 = os.path.join(RESULTS_DIR, f"convergence_plot_{FILE_TAG}.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Convergence plot written to: {p1}")

    # ── Plot 2: Per-pass timing traces ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for pol in POLICIES:
        grp = df_conv[df_conv["policy"] == pol]
        c   = COLORS.get(pol, "grey")
        ls  = LINESTYLES.get(pol, "-")
        axes[0].plot(grp["iteration"], grp["fwd_time_s"],
                     label=pol, color=c, linestyle=ls, linewidth=1.2, alpha=0.85)
        axes[1].plot(grp["iteration"], grp["bwd_time_s"],
                     label=pol, color=c, linestyle=ls, linewidth=1.2, alpha=0.85)
    for ax, title in zip(axes, ["Forward-pass time", "Backward-pass time"]):
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time (s)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    p2 = os.path.join(RESULTS_DIR, f"pass_timing_traces_{FILE_TAG}.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Pass timing traces written to: {p2}")

    # ── Plot 3: Pass timing box plots ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, col, title in zip(
        axes,
        ["fwd_time_s", "bwd_time_s"],
        ["Forward-pass time (s)", "Backward-pass time (s)"],
    ):
        data    = [df_conv.loc[df_conv["policy"] == pol, col].dropna().values
                   for pol in POLICIES]
        labels  = [p.replace(" SDDiP", "\nSDDiP") for p in POLICIES]
        colors_ = [COLORS.get(p, "grey") for p in POLICIES]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
        for patch, c in zip(bp["boxes"], colors_):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_ylabel("Seconds")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    p3 = os.path.join(RESULTS_DIR, f"pass_timing_boxplots_{FILE_TAG}.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"Pass timing boxplots written to: {p3}")

    # ── Plot 4: Per-facility open frequency heatmaps (one per policy) ─────────
    n_facilities = df_open_stat["Facility"].nunique()
    n_stages     = df_open_stat["Stage"].nunique()

    fig, axes = plt.subplots(1, len(POLICIES), figsize=(5 * len(POLICIES), 4.5))
    if len(POLICIES) == 1:
        axes = [axes]
    for ax, pol in zip(axes, POLICIES):
        sub  = df_open_stat[df_open_stat["Policy"] == pol].sort_values(
            ["Stage", "Facility"])
        mat  = sub.pivot(index="Facility", columns="Stage",
                         values="Open_Frequency").values
        im   = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="Blues",
                         aspect="auto", interpolation="nearest")
        ax.set_xticks(range(n_stages))
        ax.set_xticklabels([f"t={t+1}" for t in range(n_stages)])
        ax.set_yticks(range(n_facilities))
        ax.set_yticklabels([f"f{i+1}" for i in range(n_facilities)])
        ax.set_xlabel("Stage")
        ax.set_ylabel("Facility")
        ax.set_title(f"{pol}\nOpen frequency")
        # Annotate cells
        for row in range(n_facilities):
            for col in range(n_stages):
                val = mat[row, col]
                ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if val < 0.7 else "white")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    p4 = os.path.join(RESULTS_DIR, f"opening_freq_heatmap_{FILE_TAG}.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"Opening frequency heatmap written to: {p4}")

    # ── Plot 5: Average number of open facilities per stage ───────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for pol in POLICIES:
        sub    = df_open_summ[df_open_summ["Policy"] == pol].sort_values("Stage")
        stages = sub["Stage"].values
        avgs   = sub["Avg_N_Open"].values
        c      = COLORS.get(pol, "grey")
        ls     = LINESTYLES.get(pol, "-")
        ax.plot(stages, avgs, label=pol, color=c, linestyle=ls,
                linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Avg # open facilities")
    ax.set_title("Average open facilities per stage")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    p5 = os.path.join(RESULTS_DIR, f"avg_open_per_stage_{FILE_TAG}.png")
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"Avg open facilities plot written to: {p5}")
