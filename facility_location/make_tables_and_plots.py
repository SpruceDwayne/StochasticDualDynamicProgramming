"""
make_tables_and_plots.py

Reads all results_K{k}/{type}/ CSV files and produces:
  Table 1  (×3, one per k): performance comparison — both dist=true and dist=SAA
  Table 2  (×1):            computational effort
  Table 3  (×3, one per k): first-stage policy decisions
  Figure 1 (×3, one per k): convergence curves (LB vs iteration + LB vs wall-time)
  Figure 2 (×3, one per k): facility opening-frequency heatmaps (dist=true)

All LaTeX tables use booktabs + siunitx. All figures are saved as PDF.
Run from the facility_location/ directory (or adjust ROOT below).
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection

matplotlib.rcParams.update({
    "text.usetex": False,          # set True if you have a LaTeX install
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# ── CONFIG ────────────────────────────────────────────────────────────────────

ROOT    = os.path.dirname(os.path.abspath(__file__))
K_VALS  = [1, 2, 3]
TYPES   = ["A", "B", "C", "D"]
N_SAAS  = [25, 50]
OUT_DIR = os.path.join(ROOT, "latex_output")
os.makedirs(OUT_DIR, exist_ok=True)

POLICY_LABELS = {"DDU-SDDiP": "DDU-SDDiP", "SDDiP": "SDDiP"}
TYPE_LABELS = {
    "A": "A (all zones)",
    "B": "B (nearest zone)",
    "C": "C (nearest active)",
    "D": "D (mixed signs)",
}
COLORS = {"DDU-SDDiP": "#1f77b4", "SDDiP": "#d62728"}
LINES  = {"DDU-SDDiP": "-",       "SDDiP": "--"}

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_all_data():
    """Return three DataFrames: summary, convergence, opening_stats."""
    summ_rows, conv_rows, stat_rows = [], [], []

    for k in K_VALS:
        for itype in TYPES:
            for n in N_SAAS:
                folder = os.path.join(ROOT, f"results_K{k}", itype)

                # -- comparison_summary
                path = os.path.join(folder, f"comparison_summary_G2_{itype}_N{n}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df["k_max"] = k
                    summ_rows.append(df)

                # -- convergence_data
                path = os.path.join(folder, f"convergence_data_G2_{itype}_N{n}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df["k_max"] = k
                    df["Instance_Type"] = itype
                    df["N_SAA"] = n
                    # normalise column name (lowercase in this CSV)
                    df = df.rename(columns={"policy": "Policy"})
                    conv_rows.append(df)

                # -- opening_stats
                path = os.path.join(folder, f"opening_stats_G2_{itype}_N{n}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df["k_max"] = k
                    df["Instance_Type"] = itype
                    df["N_SAA"] = n
                    stat_rows.append(df)

    summary = pd.concat(summ_rows, ignore_index=True) if summ_rows else pd.DataFrame()
    conv    = pd.concat(conv_rows, ignore_index=True) if conv_rows else pd.DataFrame()
    stats   = pd.concat(stat_rows, ignore_index=True) if stat_rows else pd.DataFrame()
    return summary, conv, stats


# ── LATEX HELPERS ─────────────────────────────────────────────────────────────

def fmt_profit(x):
    """Format profit as integer with thousands separator."""
    return f"{x:,.0f}"

def fmt_std(x):
    return f"{x:,.0f}"

def fmt_gap(x):
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}"

def fmt_time(x):
    return f"{x:.0f}"

def fmt_iter(x):
    return f"{int(x)}"

def fmt_ratio(x):
    return f"{x:.1f}\\times"

def write_tex(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Written: {path}")


# ── TABLE 1: PERFORMANCE ──────────────────────────────────────────────────────

def make_table1_performance(summary: pd.DataFrame, k: int):
    """
    One table per k. Rows: instance type x N_SAA.
    Columns: DDU (true, SAA), SDDiP (true, SAA), Gap-true%, Gap-SAA%.
    """
    sub = summary[summary["k_max"] == k].copy()

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Out-of-sample performance comparison, $k_{\max}=" + str(k) + r"$. "
        r"Mean profit and standard deviation (Std) over 1{,}000 simulation paths, "
        r"evaluated under the true BetaBinomial distribution (True) and the SAA "
        r"scenario set (SAA). Gap\% $= 100\times(\bar\pi_{\text{DDU}}"
        r" - \bar\pi_{\text{STD}})/|\bar\pi_{\text{STD}}|$.}"
    )
    lines.append(r"\label{tab:perf_k" + str(k) + r"}")
    lines.append(r"\small")
    # 10 cols: Type | N | DDU-true mean | DDU-true std | DDU-SAA mean | DDU-SAA std
    #                   | STD-true mean | STD-true std | STD-SAA mean | STD-SAA std
    #                   | Gap-true | Gap-SAA
    lines.append(r"\begin{tabular}{ll rr rr rr rr rr}")
    lines.append(r"\toprule")
    lines.append(
        r" & & \multicolumn{4}{c}{DDU-SDDiP} & \multicolumn{4}{c}{SDDiP}"
        r" & \multicolumn{2}{c}{Gap (\%)} \\"
    )
    lines.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-12}")
    lines.append(
        r"Type & $N$ & \multicolumn{2}{c}{True} & \multicolumn{2}{c}{SAA}"
        r" & \multicolumn{2}{c}{True} & \multicolumn{2}{c}{SAA}"
        r" & True & SAA \\"
    )
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}"
                 r"\cmidrule(lr){7-8}\cmidrule(lr){9-10}")
    lines.append(r"& & Mean & Std & Mean & Std & Mean & Std & Mean & Std & & \\")
    lines.append(r"\midrule")

    prev_type = None
    for itype in TYPES:
        for n in N_SAAS:
            rows = sub[(sub["Instance_Type"] == itype) & (sub["N_SAA"] == n)]
            if rows.empty:
                continue

            def get(policy, dist):
                r = rows[(rows["Policy"] == policy) &
                         (rows["Eval_Distribution"] == dist)]
                if r.empty:
                    return None, None
                return r.iloc[0]["Avg_Profit"], r.iloc[0]["Std_Dev"]

            ddu_true_m, ddu_true_s = get("DDU-SDDiP", "true")
            ddu_saa_m,  ddu_saa_s  = get("DDU-SDDiP", "SAA")
            std_true_m, std_true_s = get("SDDiP",     "true")
            std_saa_m,  std_saa_s  = get("SDDiP",     "SAA")

            if any(v is None for v in [ddu_true_m, std_true_m, ddu_saa_m, std_saa_m]):
                continue

            gap_true = 100 * (ddu_true_m - std_true_m) / abs(std_true_m)
            gap_saa  = 100 * (ddu_saa_m  - std_saa_m)  / abs(std_saa_m)

            # add a thin rule between instance type groups
            if prev_type is not None and itype != prev_type:
                lines.append(r"\midrule")
            prev_type = itype

            type_label = itype if n == N_SAAS[0] else ""

            row = (
                f"{type_label} & {n}"
                f" & {fmt_profit(ddu_true_m)} & {fmt_std(ddu_true_s)}"
                f" & {fmt_profit(ddu_saa_m)}  & {fmt_std(ddu_saa_s)}"
                f" & {fmt_profit(std_true_m)} & {fmt_std(std_true_s)}"
                f" & {fmt_profit(std_saa_m)}  & {fmt_std(std_saa_s)}"
                f" & {fmt_gap(gap_true)} & {fmt_gap(gap_saa)}"
                r" \\"
            )
            lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = os.path.join(OUT_DIR, f"table1_performance_K{k}.tex")
    write_tex(path, "\n".join(lines))


# ── TABLE 2: COMPUTATION ──────────────────────────────────────────────────────

def make_table2_computation(summary: pd.DataFrame):
    """
    One combined table. Rows: k x instance x N_SAA.
    Columns: DDU (iters, time, s/iter) | SDDiP (iters, time, s/iter) | Overhead.
    """
    # One row per (k, instance, N_SAA, policy) — deduplicate over eval_dist
    sub = (summary.drop_duplicates(subset=["k_max", "Instance_Type", "N_SAA", "Policy"])
                  .copy())
    sub["s_per_iter"] = sub["Wall_Time_s"] / sub["Iterations"]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Computational effort. Iters: number of training iterations; "
        r"Time: total wall-clock time (seconds); s/iter: average seconds per iteration; "
        r"Overhead: DDU time / SDDiP time.}"
    )
    lines.append(r"\label{tab:computation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llr rrr rrr r}")
    lines.append(r"\toprule")
    lines.append(
        r"$k_{\max}$ & Type & $N$ "
        r"& \multicolumn{3}{c}{DDU-SDDiP} "
        r"& \multicolumn{3}{c}{SDDiP} "
        r"& Overhead \\"
    )
    lines.append(r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}")
    lines.append(
        r"& & & Iters & Time (s) & s/iter "
        r"& Iters & Time (s) & s/iter & \\"
    )
    lines.append(r"\midrule")

    prev_k = None
    for k in K_VALS:
        for itype in TYPES:
            for n in N_SAAS:
                ddu_row = sub[(sub["k_max"] == k) & (sub["Instance_Type"] == itype) &
                              (sub["N_SAA"] == n) & (sub["Policy"] == "DDU-SDDiP")]
                std_row = sub[(sub["k_max"] == k) & (sub["Instance_Type"] == itype) &
                              (sub["N_SAA"] == n) & (sub["Policy"] == "SDDiP")]
                if ddu_row.empty or std_row.empty:
                    continue

                ddu = ddu_row.iloc[0]
                std = std_row.iloc[0]
                overhead = ddu["Wall_Time_s"] / std["Wall_Time_s"]

                if prev_k is not None and k != prev_k:
                    lines.append(r"\midrule")
                prev_k = k

                k_label    = str(k) if (itype == TYPES[0] and n == N_SAAS[0]) else ""
                type_label = itype  if n == N_SAAS[0] else ""

                row = (
                    f"{k_label} & {type_label} & {n}"
                    f" & {fmt_iter(ddu['Iterations'])} & {fmt_time(ddu['Wall_Time_s'])}"
                    f" & {ddu['s_per_iter']:.1f}"
                    f" & {fmt_iter(std['Iterations'])} & {fmt_time(std['Wall_Time_s'])}"
                    f" & {std['s_per_iter']:.1f}"
                    f" & ${fmt_ratio(overhead)}$"
                    r" \\"
                )
                lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = os.path.join(OUT_DIR, "table2_computation.tex")
    write_tex(path, "\n".join(lines))


# ── TABLE 3: POLICY ───────────────────────────────────────────────────────────

def make_table3_policy(summary: pd.DataFrame, k: int):
    """
    One table per k. Rows: instance x N_SAA.
    Columns: DDU stage-1 facs | DDU region | SDDiP stage-1 facs | SDDiP region | Same?
    """
    sub = summary[summary["k_max"] == k].copy()
    sub = sub.drop_duplicates(subset=["k_max", "Instance_Type", "N_SAA", "Policy"])

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{First-stage facility decisions, $k_{\max}=" + str(k) + r"$. "
        r"Facs: set of facilities opened in stage 1; Region: resulting demand region "
        r"(index into the $2^5=32$ zone activation patterns); "
        r"Same: whether both policies open identical facilities.}"
    )
    lines.append(r"\label{tab:policy_k" + str(k) + r"}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llr cc cc c}")
    lines.append(r"\toprule")
    lines.append(
        r" & & & \multicolumn{2}{c}{DDU-SDDiP} "
        r"& \multicolumn{2}{c}{SDDiP} & \\"
    )
    lines.append(r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r"Type & $N$ & & Facilities & Region & Facilities & Region & Same? \\")
    lines.append(r"\midrule")

    prev_type = None
    for itype in TYPES:
        for n in N_SAAS:
            ddu_row = sub[(sub["Instance_Type"] == itype) & (sub["N_SAA"] == n) &
                          (sub["Policy"] == "DDU-SDDiP")]
            std_row = sub[(sub["Instance_Type"] == itype) & (sub["N_SAA"] == n) &
                          (sub["Policy"] == "SDDiP")]
            if ddu_row.empty or std_row.empty:
                continue

            ddu = ddu_row.iloc[0]
            std = std_row.iloc[0]

            same = "Yes" if ddu["First_Stage_Facs"] == std["First_Stage_Facs"] else "No"

            # clean up facility list strings like "[3, 6]" → "\{3,6\}"
            def clean_facs(s):
                nums = re.findall(r"\d+", str(s))
                return r"$\{" + ",".join(nums) + r"\}$"

            ddu_facs = clean_facs(ddu["First_Stage_Facs"])
            std_facs = clean_facs(std["First_Stage_Facs"])

            if prev_type is not None and itype != prev_type:
                lines.append(r"\midrule")
            prev_type = itype

            type_label = itype if n == N_SAAS[0] else ""

            row = (
                f"{type_label} & {n} &"
                f" & {ddu_facs} & {int(ddu['Active_Region'])}"
                f" & {std_facs} & {int(std['Active_Region'])}"
                f" & {same}"
                r" \\"
            )
            lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = os.path.join(OUT_DIR, f"table3_policy_K{k}.tex")
    write_tex(path, "\n".join(lines))


# ── FIGURE 1: CONVERGENCE ─────────────────────────────────────────────────────

def make_figure1_convergence(conv: pd.DataFrame, k: int):
    """
    2 rows × 4 cols per figure (one per k).
      Row 0: LB vs iteration
      Row 1: LB vs wall-time (s)
    Each column = one instance type A/B/C/D.
    Both policies on shared y-axis. Horizontal dashed line + annotation for final LB.
    N_SAA=50 only (cleaner; both lines per policy would clutter).
    """
    sub = conv[(conv["k_max"] == k) & (conv["N_SAA"] == 50)].copy()

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))
    fig.suptitle(f"Convergence — $k_{{\\max}}={k}$, $N=50$", y=1.01)

    for col, itype in enumerate(TYPES):
        for row, (xkey, xlabel) in enumerate([("iteration", "Iteration"),
                                               ("time_s",    "Wall time (s)")]):
            ax = axes[row, col]
            data = sub[sub["Instance_Type"] == itype]

            for policy in ["DDU-SDDiP", "SDDiP"]:
                pdata = data[data["Policy"] == policy].sort_values(xkey)
                if pdata.empty:
                    continue
                xs = pdata[xkey].values
                ys = pdata["lower_bound"].values
                ax.plot(xs, ys,
                        color=COLORS[policy],
                        linestyle=LINES[policy],
                        linewidth=1.2,
                        label=policy)
                # annotate final LB
                ax.axhline(ys[-1], color=COLORS[policy],
                           linestyle=":", linewidth=0.6, alpha=0.6)
                ax.annotate(
                    f"{ys[-1]:,.0f}",
                    xy=(xs[-1], ys[-1]),
                    xytext=(4, 2), textcoords="offset points",
                    fontsize=6, color=COLORS[policy], va="bottom",
                )

            if row == 0:
                ax.set_title(TYPE_LABELS[itype])
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel("Lower bound")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{x:,.0f}"))
            ax.tick_params(axis="y", labelsize=6)
            if row == 0 and col == 0:
                ax.legend(loc="lower right", framealpha=0.8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"fig1_convergence_K{k}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Written: {path}")


# ── FIGURE 2: FACILITY OPENING TIMELINES ─────────────────────────────────────

def make_figure2_heatmaps(stats: pd.DataFrame, k: int):
    """
    2×2 grid (A/B top, C/D bottom), dist=true, N_SAA=50.
    Each cell: facilities on y-axis, stages on x-axis.
    For each (facility, policy) a horizontal line starts at the first stage the
    facility opens and continues to the end (facilities never close).
    Line opacity at each stage = opening frequency at that stage, so the line
    builds up from faint to solid as more scenarios have opened the facility.
    DDU-SDDiP: blue (offset up); SDDiP: red (offset down).
    """
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    sub = stats[(stats["k_max"] == k) & (stats["N_SAA"] == 50) &
                (stats["Eval_Distribution"] == "true")].copy()

    n_stages     = int(sub["Stage"].max())    if not sub.empty else 4
    n_facilities = int(sub["Facility"].max()) if not sub.empty else 8

    type_grid = [["A", "B"], ["C", "D"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)


    policy_cfg = {
        "DDU-SDDiP": {"color": COLORS["DDU-SDDiP"], "y_off":  0.17, "lw": 3.5},
        "SDDiP":     {"color": COLORS["SDDiP"],     "y_off": -0.17, "lw": 3.5},
    }

    for row in range(2):
        for col in range(2):
            itype = type_grid[row][col]
            ax    = axes[row, col]

            for policy, cfg in policy_cfg.items():
                data  = sub[(sub["Policy"] == policy) & (sub["Instance_Type"] == itype)]
                r, g, b = mcolors.to_rgb(cfg["color"])

                for fi in range(1, n_facilities + 1):
                    fdata = (data[data["Facility"] == fi]
                             .sort_values("Stage")
                             .reset_index(drop=True))
                    if fdata.empty:
                        continue

                    y = fi + cfg["y_off"]

                    # Build one segment per stage where freq > 0.
                    # Each segment spans [t-0.5, t+0.5] with alpha = frequency.
                    # LineCollection lets us set per-segment RGBA independently.
                    segments, colors = [], []
                    for _, r_ in fdata.iterrows():
                        freq = float(r_["Open_Frequency"])
                        if freq <= 0:
                            continue
                        t  = int(r_["Stage"])
                        x0 = t - 0.5
                        x1 = t + 0.5
                        segments.append([(x0, y), (x1, y)])
                        colors.append((r, g, b, freq))   # RGBA

                    if segments:
                        lc = LineCollection(segments, colors=colors,
                                            linewidth=cfg["lw"],
                                            capstyle="butt")
                        ax.add_collection(lc)

            # horizontal separators between facility rows
            for fi in range(1, n_facilities + 1):
                ax.axhline(fi + 0.5, color="#aaaaaa", linewidth=0.8, zorder=0)

            # vertical dotted lines at each stage boundary
            for t in range(1, n_stages + 1):
                ax.axvline(t + 0.5, color="lightgrey", linewidth=0.4,
                           linestyle=":", zorder=0)

            ax.set_xlim(0.5, n_stages + 0.5)
            ax.set_ylim(0.5, n_facilities + 0.5)
            ax.set_xticks(range(1, n_stages + 1))
            ax.set_xticklabels([f"t={t}" for t in range(1, n_stages + 1)])
            ax.set_yticks(range(1, n_facilities + 1))
            ax.set_yticklabels([f"F{f}" for f in range(1, n_facilities + 1)])
            ax.set_title(TYPE_LABELS[itype])
            ax.set_xlabel("Stage")
            ax.set_ylabel("Facility")

    legend_handles = [
        mpatches.Patch(color=COLORS["DDU-SDDiP"], label="DDU-SDDiP"),
        mpatches.Patch(color=COLORS["SDDiP"],     label="SDDiP"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

    path = os.path.join(OUT_DIR, f"fig2_heatmaps_K{k}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Written: {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    summary, conv, stats = load_all_data()
    print(f"  summary rows: {len(summary)}, conv rows: {len(conv)}, stat rows: {len(stats)}")

    print("\nGenerating tables...")
    for k in K_VALS:
        make_table1_performance(summary, k)
        make_table3_policy(summary, k)
    make_table2_computation(summary)

    print("\nGenerating figures...")
    for k in K_VALS:
        make_figure1_convergence(conv, k)
        make_figure2_heatmaps(stats, k)

    print(f"\nAll output written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
