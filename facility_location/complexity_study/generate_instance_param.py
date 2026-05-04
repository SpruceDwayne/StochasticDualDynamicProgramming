"""
generate_instance_param.py
==========================
Parameterized facility-location instance generator for the DDU-SDDiP
complexity study.  All structural dimensions are CLI arguments; the instance
JSON is compatible with the existing Julia solver stack.

Usage
-----
    python generate_instance_param.py \
        --interaction D \
        --num-customers 6 \
        --num-facilities 16 \
        --num-zones 3 \
        --stages 4 \
        --k-budget 4 \
        --seed 123 \
        --output instances/cx_D_J6_I16_Z3_T4.json
"""

import argparse
import json
import sys
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from itertools import chain, combinations
from scipy.stats import betabinom


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def _mean_to_bb_params(mean, scale, n):
    p = float(np.clip(mean / n, 1e-6, 1 - 1e-6))
    return max(p * scale, 1e-6), max((1 - p) * scale, 1e-6)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_instance(
    interaction="D",
    num_customers=3,
    num_facilities=8,
    num_zones=3,
    T=3,
    k_budget=None,
    max_demand=5,
    C=3,
    R=500,
    O=500,
    c_transport=2,
    ALPHA_BASE=0.9,
    BETA_BASE=0.6,
    BASE_SCALE=12.0,
    seed=123,
):
    """
    Generate a DDU facility-location instance and return it as a dict.

    Parameters mirror generate_instance.py but are all keyword arguments
    so the function can be called programmatically from orchestrate.py.

    num_zones must satisfy 1 <= num_zones <= num_facilities.
    k_budget defaults to max(1, num_facilities // 4) if None.
    """
    assert 1 <= num_zones <= num_facilities, (
        f"num_zones={num_zones} must satisfy 1 <= num_zones <= num_facilities={num_facilities}"
    )
    assert num_customers >= 1 and num_facilities >= 1 and T >= 2

    if k_budget is None:
        k_budget = max(1, num_facilities // 4)

    rng = np.random.RandomState(seed)
    grid_size = 200

    # ── Customers ────────────────────────────────────────────────────────────
    mean_demands = 2.0 * np.ones(num_customers)
    alpha_beta = np.array([_mean_to_bb_params(m, BASE_SCALE, max_demand)
                           for m in mean_demands])
    alpha_vals = alpha_beta[:, 0]
    beta_vals  = alpha_beta[:, 1]
    cust_coords = rng.uniform(0, grid_size, size=(num_customers, 2))

    # ── Facilities ───────────────────────────────────────────────────────────
    # Place up to 2 facilities near the highest-demand customers; rest random.
    num_well_placed = min(2, num_customers)
    main_coords = []
    if num_well_placed > 0:
        top_idx = np.argsort(mean_demands)[-num_well_placed:]
        for idx in top_idx:
            cx, cy = cust_coords[idx]
            fx = np.clip(rng.uniform(cx - 10, cx + 10), 0, grid_size)
            fy = np.clip(rng.uniform(cy - 10, cy + 10), 0, grid_size)
            main_coords.append([fx, fy])

    num_random = num_facilities - num_well_placed
    middle_coords = rng.uniform(25, 175, size=(num_random, 2))
    fac_coords = (np.vstack([main_coords, middle_coords])
                  if main_coords else middle_coords)

    # ── Zones (K-means) ──────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=num_zones, random_state=42, n_init=10)
    kmeans.fit(fac_coords)
    facility_zones = kmeans.labels_.tolist()

    # Zone centroids (may differ from K-means centroids if zones are empty)
    zone_centroids = {}
    for z in range(num_zones):
        mask = np.array(facility_zones) == z
        if mask.any():
            zone_centroids[z] = fac_coords[mask].mean(axis=0)
        else:
            zone_centroids[z] = np.array([grid_size / 2.0, grid_size / 2.0])

    # ── Distance + profit matrices ────────────────────────────────────────────
    dist_matrix   = cdist(fac_coords, cust_coords)
    profit_matrix = R - c_transport * dist_matrix * 0.002

    # ── Activation regions (powerset of zones) ────────────────────────────────
    activation_regions = powerset(range(num_zones))   # 2^num_zones entries

    # ── Zone order per customer (by distance to zone centroid) ────────────────
    zone_order_per_customer = {}
    zone_dists_per_customer = {}
    for j in range(num_customers):
        cx, cy = cust_coords[j]
        dists = [
            (z, float(np.hypot(cx - zone_centroids[z][0],
                               cy - zone_centroids[z][1])))
            for z in range(num_zones)
        ]
        dists.sort(key=lambda x: x[1])
        zone_order_per_customer[j] = [z for z, _ in dists]
        zone_dists_per_customer[j] = {z: d for z, d in dists}

    instance = {
        "interaction_type": interaction,
        "num_customers":    num_customers,
        "num_facilities":   num_facilities,
        "num_zones":        num_zones,
        "max_demand":       max_demand,
        "R": R, "C": C, "O": O, "k": k_budget, "T": T, "c": c_transport,
        "alpha_base":  ALPHA_BASE,
        "beta_base":   BETA_BASE,
        "base_scale":  BASE_SCALE,
        "customer_coords":     cust_coords.tolist(),
        "customer_base_alpha": alpha_vals.tolist(),
        "customer_base_beta":  beta_vals.tolist(),
        "facility_coords":     fac_coords.tolist(),
        "facility_zones":      facility_zones,
        "profit_matrix":       profit_matrix.tolist(),
        "dist_matrix":         dist_matrix.tolist(),
        "zone_order_per_customer": {
            str(j): v for j, v in zone_order_per_customer.items()
        },
        "zone_dists_per_customer": {
            str(j): {str(z): d for z, d in dists.items()}
            for j, dists in zone_dists_per_customer.items()
        },
        "activation_regions": [list(r) for r in activation_regions],
    }
    return instance


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parameterized DDU facility-location instance generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--interaction",    choices=["A","B","C","D"], default="D")
    parser.add_argument("--num-customers",  type=int,   default=3)
    parser.add_argument("--num-facilities", type=int,   default=8)
    parser.add_argument("--num-zones",      type=int,   default=3,
                        help="Number of K-means zones (= log2 of region count)")
    parser.add_argument("--stages",         type=int,   default=3)
    parser.add_argument("--k-budget",       type=int,   default=None,
                        help="Max new openings per stage (default: num_facilities//4)")
    parser.add_argument("--max-demand",     type=int,   default=5)
    parser.add_argument("--capacity",       type=int,   default=3,
                        help="Capacity per facility")
    parser.add_argument("--revenue",        type=int,   default=500,
                        help="Revenue per unit demand served")
    parser.add_argument("--opening-cost",   type=int,   default=500,
                        help="Fixed cost to open one facility")
    parser.add_argument("--alpha-base",     type=float, default=0.9,
                        help="DDU mean-effect decay base")
    parser.add_argument("--beta-base",      type=float, default=0.6,
                        help="DDU scale-effect decay base")
    parser.add_argument("--base-scale",     type=float, default=12.0,
                        help="BetaBinomial base concentration parameter")
    parser.add_argument("--seed",           type=int,   default=123)
    parser.add_argument("--output",         type=str,   default=None,
                        help="JSON output path (auto-named if omitted)")
    args = parser.parse_args()

    inst = generate_instance(
        interaction    = args.interaction,
        num_customers  = args.num_customers,
        num_facilities = args.num_facilities,
        num_zones      = args.num_zones,
        T              = args.stages,
        k_budget       = args.k_budget,
        max_demand     = args.max_demand,
        C              = args.capacity,
        R              = args.revenue,
        O              = args.opening_cost,
        ALPHA_BASE     = args.alpha_base,
        BETA_BASE      = args.beta_base,
        BASE_SCALE     = args.base_scale,
        seed           = args.seed,
    )

    if args.output:
        out_path = args.output
    else:
        tag = (f"cx_{args.interaction}"
               f"_J{args.num_customers}_I{args.num_facilities}"
               f"_Z{args.num_zones}_T{args.stages}_s{args.seed}")
        out_path = f"instance_{tag}.json"

    with open(out_path, "w") as f:
        json.dump(inst, f, indent=2)

    n_regions = len(inst["activation_regions"])
    print(f"Instance written: {out_path}")
    print(f"  J={args.num_customers}  I={args.num_facilities}  "
          f"Z={args.num_zones}  regions={n_regions}  T={args.stages}  "
          f"k={inst['k']}")


if __name__ == "__main__":
    main()
