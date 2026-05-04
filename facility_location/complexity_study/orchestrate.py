"""
orchestrate.py
==============
Batch runner for the DDU-SDDiP computational complexity study.
Generates all required instances and runs run_complexity.jl for each
parameter combination across three experiments.

Experiments
-----------
  1. N_SAA sweep   — fixed small instance, vary N_SAA ∈ {10,25,50,100,200}
                     with 3 independent seeds for statistical robustness.

  2. Scale sweep   — vary (num_customers, num_facilities) and number of stages T.
                     Fixed: Z=3 (8 regions), N_SAA=25, interaction D.

  3. Region sweep  — vary num_zones (= log2 of region count) from 1 to 6.
                     Fixed: J=6, I=12, T=3, N_SAA=25, interaction D.

Usage
-----
    python orchestrate.py                        # run all experiments
    python orchestrate.py --exp 1                # only experiment 1
    python orchestrate.py --exp 2 3              # experiments 2 and 3
    python orchestrate.py --dry-run              # print plan without running
    python orchestrate.py --skip-existing        # skip already-completed runs

Output layout
-------------
    results/
      exp1_nsaa/
        instances/   (generated JSON files)
        <convergence_*.csv and summary_*.json per run>
        master_summary.csv
      exp2_scale/
        ...
      exp3_regions/
        ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent
INST_DIR    = BASE_DIR / "instances"
RESULTS_DIR = BASE_DIR / "results"
JULIA_CMD   = "julia"                           # change to full path if needed
JULIA_RUNNER = str(BASE_DIR / "run_complexity.jl")
GEN_SCRIPT   = str(BASE_DIR / "generate_instance_param.py")

INST_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment grids
# ---------------------------------------------------------------------------

# Experiment 1: N_SAA sweep
# Fixed instance: D, J=3, I=8, Z=3, T=3.  Vary N_SAA only.
# One seed suffices: per-iteration time is deterministic given N_SAA;
# the specific scenario draw does not affect the timing complexity question.
EXP1_FIXED = dict(interaction="D", num_customers=3, num_facilities=8,
                  num_zones=3, stages=3, k_budget=2)
EXP1_NSAA  = [10, 25, 50, 100, 200]
EXP1_SEED  = 123
EXP1_MAX_ITER    = 200
EXP1_TIME_BUDGET = 3600          # 1 hour per run

# Experiment 2: Scale sweep — triangular grid.
# Vary (J, I) and T.  Fixed: Z=3, N_SAA=25, interaction D.
# k_budget scales with I so the problem remains non-trivial.
# T range shrinks as instance size grows to stay within the time budget;
# the small instance extends to large T to show exponential convergence scaling.
EXP2_SIZES = [
    dict(num_customers=3,  num_facilities=8,  k_budget=2,  t_values=[3, 4, 5, 6, 7, 8, 10,12]),
    dict(num_customers=6,  num_facilities=16, k_budget=4,  t_values=[3, 4, 5, 6]),
    dict(num_customers=12, num_facilities=24, k_budget=6,  t_values=[3, 4, 5, 6]),
    #dict(num_customers=18, num_facilities=36, k_budget=9,  t_values=[3]),
    #dict(num_customers=24, num_facilities=48, k_budget=12, t_values=[3]),
]
EXP2_FIXED      = dict(interaction="D", num_zones=3)
EXP2_N_SAA      = 25
EXP2_SEED       = 123
EXP2_MAX_ITER   = 100
EXP2_TIME_BUDGET = 3600          # 60 min per run

# Experiment 3: Region sweep
# Vary num_zones from 1 to 6 (regions = 2^num_zones, from 2 to 64).
# Fixed: J=6, I=12, T=3, N_SAA=25, interaction D.
EXP3_ZONES    = [2, 3, 4, 5, 6]
EXP3_FIXED    = dict(interaction="D", num_customers=6, num_facilities=12,
                     stages=3, k_budget=3)
EXP3_N_SAA    = 25
EXP3_SEED     = 123
EXP3_MAX_ITER = 800
EXP3_TIME_BUDGET = 1800          # 30 min per run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def generate_instance(params: dict, out_path: Path, dry_run=False) -> Path:
    """Call generate_instance_param.py with the given parameters."""
    if out_path.exists():
        log(f"  instance exists, skipping generation: {out_path.name}")
        return out_path

    cmd = [sys.executable, GEN_SCRIPT, "--output", str(out_path)]
    for k, v in params.items():
        flag = "--" + k.replace("_", "-")
        cmd += [flag, str(v)]

    if dry_run:
        log(f"  [DRY] {' '.join(cmd)}")
        return out_path

    log(f"  generating: {out_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Instance generation failed for {out_path}")
    return out_path


def run_julia(inst_path: Path, N_SAA: int, max_iter: int, time_budget: float,
              seed: int, out_dir: Path, tag: str,
              dry_run=False, skip_existing=False) -> dict | None:
    """
    Call run_complexity.jl for one configuration.
    Returns the parsed summary dict on success, None on dry-run or skip.
    """
    summary_path = out_dir / f"summary_{tag}.json"

    if skip_existing and summary_path.exists():
        log(f"  [SKIP] {tag} — summary already exists")
        with open(summary_path) as f:
            return json.load(f)

    cmd = [JULIA_CMD, JULIA_RUNNER,
           str(inst_path), str(N_SAA), str(max_iter),
           str(time_budget), str(seed), str(out_dir), tag]

    if dry_run:
        log(f"  [DRY] julia run_complexity.jl ... tag={tag}")
        return None

    log(f"  running: {tag}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"  [ERROR] Julia exited with code {result.returncode} for tag={tag}")
        return None

    log(f"  done: {tag}  ({elapsed:.0f}s)")

    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def write_master_csv(records: list[dict], path: Path):
    """Write list of summary dicts to a master CSV."""
    if not records:
        return
    import csv
    keys = list(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    log(f"Master CSV written: {path} ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Experiment 1 — N_SAA sweep
# ---------------------------------------------------------------------------

def run_experiment_1(dry_run=False, skip_existing=False):
    log("=" * 60)
    log("EXPERIMENT 1: N_SAA sweep")
    log("=" * 60)

    exp_dir = RESULTS_DIR / "exp1_nsaa"
    exp_dir.mkdir(parents=True, exist_ok=True)

    inst_params = EXP1_FIXED.copy()
    inst_path   = INST_DIR / "exp1_base.json"
    generate_instance(inst_params, inst_path, dry_run=dry_run)

    records = []
    for N_SAA in EXP1_NSAA:
        tag = f"exp1_N{N_SAA}"
        summary = run_julia(
            inst_path    = inst_path,
            N_SAA        = N_SAA,
            max_iter     = EXP1_MAX_ITER,
            time_budget  = EXP1_TIME_BUDGET,
            seed         = EXP1_SEED,
            out_dir      = exp_dir,
            tag          = tag,
            dry_run      = dry_run,
            skip_existing= skip_existing,
        )
        if summary:
            summary["exp"] = 1
            summary["N_SAA_var"] = N_SAA
            records.append(summary)

    if not dry_run:
        write_master_csv(records, exp_dir / "master_summary.csv")


# ---------------------------------------------------------------------------
# Experiment 2 — Scale sweep
# ---------------------------------------------------------------------------

def run_experiment_2(dry_run=False, skip_existing=False):
    log("=" * 60)
    log("EXPERIMENT 2: Scale sweep (J, I, T)")
    log("=" * 60)

    exp_dir = RESULTS_DIR / "exp2_scale"
    exp_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for size in EXP2_SIZES:
        J  = size["num_customers"]
        I_ = size["num_facilities"]
        k  = size["k_budget"]

        for T in size["t_values"]:
            inst_params = {k: v for k, v in {**EXP2_FIXED, **size, "stages": T}.items()
                           if k != "t_values"}
            inst_name   = f"exp2_J{J}_I{I_}_T{T}"
            inst_path   = INST_DIR / f"{inst_name}.json"

            generate_instance(inst_params, inst_path, dry_run=dry_run)

            tag = f"{inst_name}_N{EXP2_N_SAA}"
            summary = run_julia(
                inst_path    = inst_path,
                N_SAA        = EXP2_N_SAA,
                max_iter     = EXP2_MAX_ITER,
                time_budget  = EXP2_TIME_BUDGET,
                seed         = EXP2_SEED,
                out_dir      = exp_dir,
                tag          = tag,
                dry_run      = dry_run,
                skip_existing= skip_existing,
            )
            if summary:
                summary["exp"]          = 2
                summary["J"]            = J
                summary["I"]            = I_
                summary["T_var"]        = T
                summary["problem_size"] = J * I_
                records.append(summary)

    if not dry_run:
        write_master_csv(records, exp_dir / "master_summary.csv")


# ---------------------------------------------------------------------------
# Experiment 3 — Region sweep
# ---------------------------------------------------------------------------

def run_experiment_3(dry_run=False, skip_existing=False):
    log("=" * 60)
    log("EXPERIMENT 3: Region sweep (num_zones)")
    log("=" * 60)

    exp_dir = RESULTS_DIR / "exp3_regions"
    exp_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for Z in EXP3_ZONES:
        n_regions = 2 ** Z
        inst_params = {**EXP3_FIXED, "num_zones": Z}
        inst_name   = f"exp3_Z{Z}"
        inst_path   = INST_DIR / f"{inst_name}.json"

        generate_instance(inst_params, inst_path, dry_run=dry_run)

        tag = f"{inst_name}_N{EXP3_N_SAA}"
        summary = run_julia(
            inst_path    = inst_path,
            N_SAA        = EXP3_N_SAA,
            max_iter     = EXP3_MAX_ITER,
            time_budget  = EXP3_TIME_BUDGET,
            seed         = EXP3_SEED,
            out_dir      = exp_dir,
            tag          = tag,
            dry_run      = dry_run,
            skip_existing= skip_existing,
        )
        if summary:
            summary["exp"]       = 3
            summary["Z_var"]     = Z
            summary["n_regions"] = n_regions
            records.append(summary)

    if not dry_run:
        write_master_csv(records, exp_dir / "master_summary.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for the DDU-SDDiP complexity study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--exp", nargs="*", type=int, choices=[1, 2, 3],
                        default=[1, 2, 3],
                        help="Which experiments to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run plan without executing anything")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose summary JSON already exists")
    args = parser.parse_args()

    dry_run       = args.dry_run
    skip_existing = args.skip_existing

    if dry_run:
        log("DRY RUN — no Julia processes will be launched")

    t_start = time.time()

    if 1 in args.exp:
        run_experiment_1(dry_run=dry_run, skip_existing=skip_existing)
    if 2 in args.exp:
        run_experiment_2(dry_run=dry_run, skip_existing=skip_existing)
    if 3 in args.exp:
        run_experiment_3(dry_run=dry_run, skip_existing=skip_existing)

    elapsed = time.time() - t_start
    log(f"All experiments finished in {elapsed/3600:.2f}h")


if __name__ == "__main__":
    main()
