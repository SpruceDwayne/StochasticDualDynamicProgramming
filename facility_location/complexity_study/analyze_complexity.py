"""
analyze_complexity.py
=====================
Reads the results produced by orchestrate.py and generates
publication-quality figures for the three complexity experiments.

Usage
-----
    python analyze_complexity.py               # all experiments
    python analyze_complexity.py --exp 1       # only experiment 1
    python analyze_complexity.py --out figures # custom output directory

Output files (in --out directory, default: results/figures/)
-------------------------------------------------------------
  exp1_lb_vs_iter.pdf/png    LB convergence by iteration, one curve per N_SAA
  exp1_lb_vs_time.pdf/png    LB convergence by wall time
  exp1_iter_time.pdf/png     Mean per-iteration time vs N_SAA (log-log)
  exp2_heatmap.pdf/png       Heatmap of per-iteration time over (size, T)
  exp2_loglog.pdf/png        Log-log scatter: effective problem size vs time
  exp3_iter_time.pdf/png     Per-iteration time vs number of regions
  exp3_cuts.pdf/png          Total cuts at budget vs number of regions
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Okabe-Ito palette: perceptually uniform, colorblind-safe, 8 colours.
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # purple
    "#000000",  # black
]

# Journal-ready defaults
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "lines.linewidth":  1.4,
    "lines.markersize": 4,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
})

# Figure widths matching common journal column widths (inches)
W_SINGLE = 3.35    # single column
W_DOUBLE = 6.85    # double column
H_DEFAULT = 2.5    # default height

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_convergence(exp_dir: Path, tag: str) -> pd.DataFrame | None:
    p = exp_dir / f"convergence_{tag}.csv"
    if not p.exists():
        warnings.warn(f"Missing: {p}")
        return None
    df = pd.read_csv(p)
    df["tag"] = tag
    return df


def load_summary(exp_dir: Path, tag: str) -> dict | None:
    p = exp_dir / f"summary_{tag}.json"
    if not p.exists():
        warnings.warn(f"Missing: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def load_master(exp_dir: Path) -> pd.DataFrame | None:
    p = exp_dir / "master_summary.csv"
    if not p.exists():
        warnings.warn(f"No master_summary.csv in {exp_dir}")
        return None
    return pd.read_csv(p)


def savefig(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path)
    print(f"  saved: {path}")
    plt.close(fig)


def _loglabel(n):
    """Format integer as log2-style label: e.g. 8 → '8 (2³)'."""
    import math
    if n > 0 and (n & (n - 1)) == 0:
        exp = int(math.log2(n))
        return f"$2^{{{exp}}}$ = {n}"
    return str(n)


# ---------------------------------------------------------------------------
# Experiment 1 - N_SAA sweep
# ---------------------------------------------------------------------------

def analyze_exp1(out_dir: Path):
    """
    Produce three figures for the N_SAA sweep:
      (a) LB vs. iteration - convergence speed by N_SAA
      (b) LB vs. wall time - computational cost vs. quality
      (c) Mean per-iteration time vs. N_SAA (log-log)
    """
    from orchestrate import EXP1_NSAA

    exp_dir = RESULTS_DIR / "exp1_nsaa"
    if not exp_dir.exists():
        print("Experiment 1 results not found - skipping.")
        return

    # ── Load convergence data (one run per N_SAA) ─────────────────────────────
    nsaa_vals = sorted(EXP1_NSAA)
    colors    = {N: OKABE_ITO[i % len(OKABE_ITO)] for i, N in enumerate(nsaa_vals)}

    conv_data: dict[int, pd.DataFrame | None] = {}
    for N in nsaa_vals:
        conv_data[N] = load_convergence(exp_dir, f"exp1_N{N}")

    # ── Figure 1a: LB vs. iteration ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(W_DOUBLE, H_DEFAULT))
    for N in nsaa_vals:
        df = conv_data[N]
        if df is None:
            continue
        ax.plot(df["iteration"], df["lb"], color=colors[N], label=f"$N = {N}$")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Lower bound")
    ax.set_title("(a) LB convergence by iteration")
    ax.legend(title="$N_{\\mathrm{SAA}}$", loc="lower right", ncol=2)
    fig.tight_layout()
    savefig(fig, out_dir, "exp1_lb_vs_iter")

    # ── Figure 1b: LB vs. wall time ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(W_DOUBLE, H_DEFAULT))
    for N in nsaa_vals:
        df = conv_data[N]
        if df is None:
            continue
        ax.plot(df["time_s"], df["lb"], color=colors[N], label=f"$N = {N}$")

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Lower bound")
    ax.set_title("(b) LB convergence by wall time")
    ax.legend(title="$N_{\\mathrm{SAA}}$", loc="lower right", ncol=2)
    fig.tight_layout()
    savefig(fig, out_dir, "exp1_lb_vs_time")

    # ── Figure 1c: Violin + scatter of per-iteration times vs. N_SAA ─────────
    # Pool raw iter_time_s across all seeds; drop iteration 1 (JIT warmup).
    iter_times: dict[int, list] = {}
    for N in nsaa_vals:
        times: list[float] = []
        seed_files = sorted(exp_dir.glob(f"convergence_exp1_N{N}_s*.csv"))
        if not seed_files:
            seed_files = [exp_dir / f"convergence_exp1_N{N}.csv"]
        for path in seed_files:
            if path.exists():
                df = pd.read_csv(path)
                if "iter_time_s" in df.columns:
                    times.extend(df["iter_time_s"].iloc[1:].tolist())
        iter_times[N] = times

    valid_N = [N for N in nsaa_vals if iter_times[N]]
    if valid_N:
        fig, ax = plt.subplots(figsize=(W_SINGLE, H_DEFAULT))
        positions = list(range(len(valid_N)))
        data      = [iter_times[N] for N in valid_N]

        parts = ax.violinplot(data, positions=positions,
                              showmedians=True, showextrema=False)
        for body, N in zip(parts["bodies"], valid_N):
            body.set_facecolor(colors[N])
            body.set_alpha(0.45)
        parts["cmedians"].set_colors("black")
        parts["cmedians"].set_linewidth(1.5)

        rng = np.random.default_rng(0)
        for i, N in enumerate(valid_N):
            ys = np.array(iter_times[N])
            xs = rng.normal(i, 0.06, size=len(ys))
            ax.scatter(xs, ys, color=colors[N], s=5, alpha=0.4, zorder=3,
                       linewidths=0)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"$N={N}$" for N in valid_N])
        ax.set_xlabel("$N_{\\mathrm{SAA}}$")
        ax.set_ylabel("Per-iteration time (s)")
        ax.set_title("(c) Per-iteration cost vs. $N_{\\mathrm{SAA}}$")
        fig.tight_layout()
        savefig(fig, out_dir, "exp1_iter_time")

    print("Experiment 1 figures done.")


# ---------------------------------------------------------------------------
# Experiment 2 - Scale sweep
# ---------------------------------------------------------------------------

def analyze_exp2(out_dir: Path):
    """
    Produce two figures for the scale sweep:
      (a) Heatmap: mean per-iteration time over (J*I, T)
      (b) Log-log scatter: J*I vs per-iteration time with OLS fit
    """
    exp_dir = RESULTS_DIR / "exp2_scale"
    master  = load_master(exp_dir)
    if master is None:
        print("Experiment 2 results not found - skipping.")
        return

    # Columns we need
    required = {"nJ", "nI", "T", "mean_iter_time_s"}
    if not required.issubset(master.columns):
        print(f"master_summary.csv missing columns: {required - set(master.columns)}")
        return

    master = master.dropna(subset=["mean_iter_time_s"])
    master["size_pair"] = list(zip(master["nJ"].astype(int), master["nI"].astype(int)))
    # Use strings as index keys to avoid pandas interpreting tuples as multi-index
    master["size_label"] = [f"({j}, {i})" for j, i in master["size_pair"]]

    # ── Figure 2a: Heatmap ────────────────────────────────────────────────────
    size_pairs  = sorted(set(master["size_pair"]))  # sorted lexicographically by (J, I)
    size_labels = [f"({j}, {i})" for j, i in size_pairs]
    T_vals      = sorted(master["T"].unique())
    heat        = pd.DataFrame(index=size_labels, columns=T_vals, dtype=float)
    for _, row in master.iterrows():
        heat.loc[row["size_label"], int(row["T"])] = row["mean_iter_time_s"]

    fig, ax = plt.subplots(figsize=(W_SINGLE + 0.8, H_DEFAULT + 0.5))
    data_arr = heat.values.astype(float)
    im = ax.imshow(data_arr, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(len(T_vals)))
    ax.set_xticklabels([f"$T={t}$" for t in T_vals])
    ax.set_yticks(range(len(size_labels)))
    ax.set_yticklabels([f"$({j},\\ {i})$" for j, i in size_pairs])
    ax.set_xlabel("Stages $T$")
    ax.set_ylabel("Problem size $(J, I)$")
    ax.set_title("(a) Mean per-iteration time (s)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Time (s)")
    # Annotate cells with actual values
    for i, lbl in enumerate(size_labels):
        for j, T in enumerate(T_vals):
            val = heat.loc[lbl, T]
            if not np.isnan(val):
                txt = f"{val:.1f}" if val < 100 else f"{val:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="black" if val < 30 else "white")
    fig.tight_layout()
    savefig(fig, out_dir, "exp2_heatmap")

    # ── Figure 2b: Scatter of per-iteration time vs. problem size ────────────
    fig, ax = plt.subplots(figsize=(W_SINGLE, H_DEFAULT))
    palette    = {T: OKABE_ITO[i] for i, T in enumerate(sorted(T_vals))}
    pair_to_x  = {sp: k for k, sp in enumerate(size_pairs)}
    x_labels   = [f"$({j},\\ {i})$" for j, i in size_pairs]

    for T in sorted(T_vals):
        sub = master[master["T"] == T].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("size_pair")
        xs  = [pair_to_x[sp] for sp in sub["size_pair"]]
        ax.scatter(xs, sub["mean_iter_time_s"],
                   color=palette[T], label=f"$T={T}$", zorder=3)
        ax.plot(xs, sub["mean_iter_time_s"],
                color=palette[T], alpha=0.5, zorder=2)

    ax.set_xticks(range(len(size_pairs)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Problem size $(J, I)$")
    ax.set_ylabel("Mean iteration time (s)")
    ax.set_title("(b) Per-iteration time vs.\\ problem size")
    ax.legend(loc="upper left")
    fig.tight_layout()
    savefig(fig, out_dir, "exp2_scatter")

    # ── Figure 2c: Normalised LB vs. wall time for J×I=24 (J=3, I=8) ─────────
    # J×I = 3×8 = 24; vary T. Each curve normalised to [0,1] so convergence
    # speeds are directly comparable across different LB scales.
    target_J, target_I = 3, 8
    t_vals_24 = sorted(master[(master["nJ"] == target_J) &
                               (master["nI"] == target_I)]["T"].unique())
    palette_T  = {T: OKABE_ITO[i % len(OKABE_ITO)] for i, T in enumerate(t_vals_24)}
    n_saa_2    = int(master["N_SAA"].iloc[0]) if "N_SAA" in master.columns else 25

    fig, ax = plt.subplots(figsize=(W_DOUBLE, H_DEFAULT))
    for T in t_vals_24:
        tag  = f"exp2_J{target_J}_I{target_I}_T{T}_N{n_saa_2}"
        conv = load_convergence(exp_dir, tag)
        if conv is None:
            continue
        lb   = conv["lb"].values
        lb_min, lb_max = lb[0], lb[-1]
        if abs(lb_max - lb_min) < 1e-10:
            continue
        lb_norm = (lb - lb_min) / (lb_max - lb_min)
        ax.plot(conv["time_s"], lb_norm, color=palette_T[T], label=f"$T={T}$")

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Normalised lower bound")
    ax.set_title(f"(c) LB convergence - $(J, I)=({target_J}, {target_I})$, varying $T$")
    ax.legend(title="Stages", loc="lower right", ncol=2)
    fig.tight_layout()
    savefig(fig, out_dir, "exp2_lb_convergence")

    print("Experiment 2 figures done.")


# ---------------------------------------------------------------------------
# Experiment 3 - Region sweep
# ---------------------------------------------------------------------------

def analyze_exp3(out_dir: Path):
    """
    Produce two figures for the region sweep:
      (a) Per-iteration time vs. number of DDU regions
      (b) Total cuts at budget exhaustion vs. number of regions
    """
    exp_dir = RESULTS_DIR / "exp3_regions"
    master  = load_master(exp_dir)
    if master is None:
        print("Experiment 3 results not found - skipping.")
        return

    required = {"n_regions", "mean_iter_time_s"}
    if not required.issubset(master.columns):
        print(f"master_summary.csv missing columns: {required - set(master.columns)}")
        return

    master = master.dropna(subset=["mean_iter_time_s"]).sort_values("n_regions")
    regions = master["n_regions"].values
    mean_t  = master["mean_iter_time_s"].values
    std_t   = master.get("std_iter_time_s", pd.Series(np.zeros(len(master)))).values

    # ── Figure 3a: Per-iteration time vs. regions ─────────────────────────────
    fig, ax = plt.subplots(figsize=(W_SINGLE + 0.5, H_DEFAULT))
    ax.errorbar(regions, mean_t, yerr=std_t,
                fmt="o-", color=OKABE_ITO[4], capsize=3)
    ax.set_xlabel("Number of DDU regions $|\\mathcal{D}|$")
    ax.set_ylabel("Mean iteration time (s)")
    ax.set_title("(a) Per-iteration cost vs.\ region count")
    fig.tight_layout()
    savefig(fig, out_dir, "exp3_iter_time")

    # ── Figure 3b: Iterations completed within budget vs. regions ────────────
    # total_iters comes directly from the summary JSON via master_summary.csv.
    if "total_iters" in master.columns:
        fig, ax = plt.subplots(figsize=(W_SINGLE + 0.5, H_DEFAULT))
        ax.plot(regions, master["total_iters"].values,
                "s-", color=OKABE_ITO[5])
        ax.set_xlabel("Number of DDU regions $|\\mathcal{D}|$")
        ax.set_ylabel("Iterations completed")
        ax.set_title("(b) Iterations  vs.\ region count")
        fig.tight_layout()
        savefig(fig, out_dir, "exp3_iterations")

    # ── Figure 3c: Normalised LB vs. wall time, one curve per region count ────
    # Each curve normalised to [0,1]: lb_norm=(lb-lb_first)/(lb_last-lb_first).
    palette_r = {r: OKABE_ITO[i % len(OKABE_ITO)]
                 for i, r in enumerate(sorted(int(r) for r in regions))}

    fig, ax = plt.subplots(figsize=(W_DOUBLE, H_DEFAULT))
    n_saa = int(master["N_SAA"].iloc[0]) if "N_SAA" in master.columns else EXP3_N_SAA
    for _, row in master.iterrows():
        Z    = int(np.log2(row["n_regions"]))
        n_r  = int(row["n_regions"])
        tag  = f"exp3_Z{Z}_N{n_saa}"
        conv = load_convergence(exp_dir, tag)
        if conv is None:
            continue
        lb    = conv["lb"].values
        lb_min, lb_max = lb[0], lb[-1]
        if abs(lb_max - lb_min) < 1e-10:
            continue
        lb_norm = (lb - lb_min) / (lb_max - lb_min)
        ax.plot(conv["time_s"], lb_norm,
                color=palette_r[n_r], label=f"$|\\mathcal{{D}}|={n_r}$")

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Normalised lower bound")
    ax.set_title("(c) LB convergence - varying region count")
    ax.legend(title="Regions", loc="lower right", ncol=2)
    fig.tight_layout()
    savefig(fig, out_dir, "exp3_lb_convergence")

    print("Experiment 3 figures done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Produce publication figures for the DDU-SDDiP complexity study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--exp", nargs="*", type=int, choices=[1, 2, 3],
                        default=[1, 2, 3])
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for figures (default: results/figures)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if 1 in args.exp:
        analyze_exp1(out_dir)
    if 2 in args.exp:
        analyze_exp2(out_dir)
    if 3 in args.exp:
        analyze_exp3(out_dir)

    print(f"\nAll figures written to: {out_dir}")


if __name__ == "__main__":
    main()
