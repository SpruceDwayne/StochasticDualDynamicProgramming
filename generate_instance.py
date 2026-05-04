"""
generate_instance.py
====================
Generates a facility-location instance with DDU (decision-dependent uncertainty)
demand distributions.  The DDU interaction type is selected via --interaction.

Usage
-----
    python generate_instance.py --interaction A
    python generate_instance.py --interaction B
    python generate_instance.py --interaction C
    python generate_instance.py --interaction D

Interaction types (from the DDU literature)
--------------------------------------------
A  All zones contribute to mean and scale; contribution decays exponentially
   with distance rank (alpha_nj = 0.5^n, beta_nj = 0.4^n).
   Demand is influenced by both proximity and density of open zones.

B  Only the nearest zone z^(1)(j) influences customer j.
   Equivalent to Type A with alpha_nj = beta_nj = 0 for n >= 2.

C  Only the nearest *open* zone influences customer j, but the effect
   magnitude depends on how far that zone is (0.5^rank, 0.4^rank).

D  Nearest zone boosts mean and tightens the distribution (s(1)=0).
   Far zones cannibalize mean and loosen the distribution (s(n)=1 for n>=2).
   Mimics demand reduction when facilities are located far from a customer.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from itertools import chain, combinations
from scipy.stats import betabinom
import json
import argparse

# ===================================================================
# Parse arguments
# ===================================================================
parser = argparse.ArgumentParser(
    description='Generate DDU facility-location instance.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__
)
parser.add_argument(
    '--interaction', type=str, choices=['A', 'B', 'C', 'D'], default='B',
    help='DDU interaction type (default: A)'
)
args = parser.parse_args()
ITYPE = args.interaction
print(f"DDU interaction type: {ITYPE}")

np.random.seed(123)

# ===================================================================
# Instance parameters
# ===================================================================
num_customers  = 3
num_facilities = 8
grid_size      = 200
num_zones      = 3
max_demand     = 5     # beta-binomial support {0, ..., max_demand}

# Economic parameters
R        = 500   # revenue per unit demand served
C        = 3    # capacity per facility. The higher the capacity the less facilities will be opened by
O        = 500  # fixed opening cost
k_budget = 3    # max new openings per stage
T        = 3     # number of stages
c_transport = 2  # transportation cost per unit distance

# DDU parameters (article notation)
#   alpha_nj = ALPHA_BASE ^ n   (mean effect at zone rank n)
#   beta_nj  = BETA_BASE  ^ n   (scale effect at zone rank n)
ALPHA_BASE  = 0.9#0.7#0.5
BETA_BASE   = 0.6#0.4
BASE_SCALE  = 12 #6   # base concentration for beta-binomial

# ===================================================================
# 1. Generate customers
# ===================================================================
#mean_demands = np.linspace(1.5, 5, num_customers)
#mean_demands = np.linspace(0.5, max_demand - 0.5, num_customers)
mean_demands = 2*np.ones(num_customers)

def mean_to_bb_params(mean, scale=BASE_SCALE, n=max_demand):
    p = mean / n
    return p * scale, (1 - p) * scale

alpha_beta       = np.array([mean_to_bb_params(m) for m in mean_demands])
alpha_vals       = alpha_beta[:, 0]
beta_vals        = alpha_beta[:, 1]
cust_coords      = np.random.uniform(0, grid_size, size=(num_customers, 2))
cust_mean_demands = max_demand * (alpha_vals / (alpha_vals + beta_vals))

customers = pd.DataFrame({
    'x': cust_coords[:, 0],
    'y': cust_coords[:, 1],
    'mean_demand': cust_mean_demands,
    'alpha': alpha_vals,
    'beta':  beta_vals
})

# ===================================================================
# 2. Generate facilities
# ===================================================================
num_well_placed = 2
main_fac_coords = []
if num_well_placed > 0:
    largest_idx = np.argsort(customers['mean_demand'].values)[-num_well_placed:]
    for idx in largest_idx:
        cx, cy = customers.loc[idx, ['x', 'y']]
        fx = np.clip(np.random.uniform(cx - 10, cx + 10), 0, grid_size)
        fy = np.clip(np.random.uniform(cy - 10, cy + 10), 0, grid_size)
        main_fac_coords.append([fx, fy])

num_random = num_facilities - num_well_placed
middle_fac_coords = np.random.uniform(25, 75, size=(num_random, 2))
facility_coords   = np.vstack([main_fac_coords, middle_fac_coords]) if main_fac_coords else middle_fac_coords

facilities = pd.DataFrame(facility_coords, columns=['x', 'y'])
facilities['capacity']     = C
facilities['opening_cost'] = O

# ===================================================================
# 3. Zones via K-means clustering of facilities
# ===================================================================
kmeans = KMeans(n_clusters=num_zones, random_state=42, n_init=10).fit(facility_coords)
facilities['zone'] = kmeans.labels_

# ===================================================================
# 4. Distance matrix and per-unit profit
# ===================================================================
dist_matrix   = cdist(facility_coords, cust_coords)
profit_matrix = R - c_transport * dist_matrix * 0.002

# ===================================================================
# 5. Activation regions (powerset of zones)
# ===================================================================
def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

activation_regions = powerset(range(num_zones))

print(f"Number of activation regions: {len(activation_regions)}")

# ===================================================================
# 6. Zone order per customer (by distance to zone centroid)
# ===================================================================
zone_centroids = facilities.groupby('zone')[['x', 'y']].mean().to_dict('index')

zone_order_per_customer = {}   # {j: [z_rank1, z_rank2, ...]}
zone_dists_per_customer = {}   # {j: {z: distance_to_centroid}}

for j in range(num_customers):
    cx, cy = customers.loc[j, ['x', 'y']]
    dists = []
    for z in range(num_zones):
        zx, zy = zone_centroids[z]['x'], zone_centroids[z]['y']
        dists.append((z, np.hypot(cx - zx, cy - zy)))
    dists_sorted = sorted(dists, key=lambda x: x[1])
    zone_order_per_customer[j] = [z for z, _ in dists_sorted]
    zone_dists_per_customer[j] = dict(dists)

# ===================================================================
# 7. DDU demand distributions
# ===================================================================

def _zone_active_vector(x_open):
    zone_active = np.zeros(num_zones, dtype=int)
    for i in range(num_facilities):
        if x_open[i] == 1:
            zone_active[facilities.loc[i, 'zone']] = 1
    return zone_active


def get_customer_distributions(x_open, itype=None, n=max_demand):
    """
    Return {j: probability_array} for all customers given facility openings x_open.

    The DDU effect modifies each customer's demand mean and distribution scale
    based on which zones are active.  The sign convention uses:
        sign(n) = (1 - 2 * s(n))
    where s(n) = 0 for all types except D where s(n>=2) = 1.

    Mean   : mean_j  = base_mean_j  * (1 + sum_n  alpha_nj * sign(n) * active_n)
    Scale  : scale_j = BASE_SCALE   * (1 + sum_n  beta_nj  * sign(n) * active_n)
      Higher scale → tighter distribution (lower variance).
    """
    if itype is None:
        itype = ITYPE

    zone_active  = _zone_active_vector(x_open)
    distributions = {}

    for j in range(num_customers):
        base_mean  = customers.loc[j, 'mean_demand']
        zone_order = zone_order_per_customer[j]

        mean_delta  = 0.0
        scale_delta = 0.0

        if itype == 'A':
            # All zones contribute; weight decays exponentially with rank.
            for n_idx, z in enumerate(zone_order):
                rank      = n_idx + 1
                alpha_nj  = ALPHA_BASE ** rank
                beta_nj   = BETA_BASE  ** rank
                active    = zone_active[z]
                mean_delta  += alpha_nj * active
                scale_delta += beta_nj  * active

        elif itype == 'B':
            # Only the nearest zone (rank 1) matters.
            near_zone   = zone_order[0]
            active      = zone_active[near_zone]
            mean_delta  = ALPHA_BASE * active        # 0.5^1
            scale_delta = BETA_BASE  * active        # 0.4^1

        elif itype == 'C':
            # Only the nearest *open* zone matters;
            # effect magnitude = 0.5^(rank of that zone), 0.4^(rank of that zone).
            for n_idx, z in enumerate(zone_order):
                if zone_active[z] == 1:
                    rank        = n_idx + 1
                    mean_delta  = ALPHA_BASE ** rank
                    scale_delta = BETA_BASE  ** rank
                    break          # stop at the first open zone

        elif itype == 'D':
            # s(1)=0  → nearest zone: positive sign → higher mean, tighter dist
            # s(n>=2)=1 → far zones:  negative sign → lower  mean, looser  dist
            for n_idx, z in enumerate(zone_order):
                rank     = n_idx + 1
                alpha_nj = ALPHA_BASE ** rank
                beta_nj  = BETA_BASE  ** rank
                s_n      = 0 if rank == 1 else 1
                sign     = 1 - 2 * s_n          # +1 for rank 1, -1 for rank >= 2
                active   = zone_active[z]
                mean_delta  += alpha_nj * sign * active
                scale_delta += beta_nj  * sign * active

        mean  = base_mean  * (1 + mean_delta)
        scale = BASE_SCALE * (1 + scale_delta)

        mean  = float(np.clip(mean,  1, n - 1e-6))
        scale = max(scale, 0.5)

        p = mean / n
        a = max(p * scale,       1e-6)
        b = max((1 - p) * scale, 1e-6)

        k_vals = np.arange(n + 1)
        distributions[j] = betabinom.pmf(k_vals, n, a, b)

    return distributions


def expected_demand(dist_dict, j):
    return float(np.dot(np.arange(max_demand + 1), dist_dict[j]))

# ===================================================================
# 8. Sanity check
# ===================================================================
print("\n" + "=" * 60)
print(f"ECONOMIC SANITY CHECK  (type {ITYPE})")
print("=" * 60)

x_none = np.zeros(num_facilities, dtype=int)
x_all  = np.ones(num_facilities,  dtype=int)

dists_none = get_customer_distributions(x_none)
dists_all  = get_customer_distributions(x_all)

total_ed_none = sum(expected_demand(dists_none, j) for j in range(num_customers))
total_ed_all  = sum(expected_demand(dists_all,  j) for j in range(num_customers))

print(f"\nParameters: R={R}, C={C}, O={O}, k={k_budget}, T={T}")
print(f"DDU type: {ITYPE}  |  alpha_base={ALPHA_BASE}, beta_base={BETA_BASE}")
print(f"Max demand per customer: {max_demand}")
print(f"Total capacity (all open): {num_facilities * C}")
print(f"\nNo facilities open:   total expected demand = {total_ed_none:.2f}")
print(f"All facilities open:  total expected demand = {total_ed_all:.2f}")

print(f"\n{'Cust':>4} {'Base':>6} {'No fac':>7} {'All fac':>8} {'Change':>9}")
for j in range(num_customers):
    ed_none = expected_demand(dists_none, j)
    ed_all  = expected_demand(dists_all,  j)
    pct     = (ed_all - ed_none) / ed_none * 100
    print(f"  {j:>2} {customers.loc[j,'mean_demand']:>6.2f} {ed_none:>7.2f} {ed_all:>8.2f} {pct:>+8.1f}%")

# ===================================================================
# 9. DDU effect table for representative customers
# ===================================================================
print("\n" + "=" * 60)
print(f"DDU SUBSTITUTION EFFECT  (type {ITYPE})")
print("=" * 60)

region_labels = [("none", ())] + [
    ("{" + ",".join(str(z) for z in subset) + "}", subset)
    for r in range(1, num_zones + 1)
    for subset in combinations(range(num_zones), r)
]

for j_show in [0, 2]:
    zo = zone_order_per_customer[j_show]
    print(f"\n  Customer {j_show} | base mean={customers.loc[j_show,'mean_demand']:.2f}"
          f" | zone order (near→far): {zo}")
    print(f"  {'Active zones':<12} {'mean':>8} {'scale':>7} {'vs base':>10}")
    print("  " + "-" * 44)

    base = customers.loc[j_show, 'mean_demand']
    for label, active_zones in region_labels:
        x_pat = np.zeros(num_facilities, dtype=int)
        for z in active_zones:
            for i in facilities[facilities['zone'] == z].index:
                x_pat[i] = 1
                break
        zone_active = _zone_active_vector(x_pat)

        # Replicate the per-customer computation to show mean & scale
        mean_delta = scale_delta = 0.0
        zo_j = zone_order_per_customer[j_show]

        if ITYPE == 'A':
            for n_idx, z in enumerate(zo_j):
                rank = n_idx + 1
                mean_delta  += ALPHA_BASE ** rank * zone_active[z]
                scale_delta += BETA_BASE  ** rank * zone_active[z]
        elif ITYPE == 'B':
            active = zone_active[zo_j[0]]
            mean_delta, scale_delta = ALPHA_BASE * active, BETA_BASE * active
        elif ITYPE == 'C':
            for n_idx, z in enumerate(zo_j):
                if zone_active[z] == 1:
                    rank = n_idx + 1
                    mean_delta, scale_delta = ALPHA_BASE ** rank, BETA_BASE ** rank
                    break
        elif ITYPE == 'D':
            for n_idx, z in enumerate(zo_j):
                rank = n_idx + 1
                sign = 1 if rank == 1 else -1
                mean_delta  += ALPHA_BASE ** rank * sign * zone_active[z]
                scale_delta += BETA_BASE  ** rank * sign * zone_active[z]

        mean_shown  = base * (1 + mean_delta)
        scale_shown = BASE_SCALE * (1 + scale_delta)
        pct = (mean_shown - base) / base * 100
        print(f"  {label:<12} {mean_shown:>8.2f} {scale_shown:>7.2f} {pct:>+9.1f}%")

# ===================================================================
# 10. Visualization
# ===================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
zone_palette = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0", "#FF9800"]

# --- Panel 1: Layout ---
ax  = axes[0]
xx, yy = np.meshgrid(np.linspace(0, grid_size, 300),
                     np.linspace(0, grid_size, 300))
zone_pred = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax.contourf(xx, yy, zone_pred,
            levels=[-0.5 + i for i in range(num_zones + 1)],
            cmap=ListedColormap(zone_palette), alpha=0.25)
ax.scatter(customers['x'], customers['y'],
           s=customers['mean_demand'] * 40 + 20,
           c='steelblue', alpha=0.8, edgecolors='navy', zorder=3, label='Customers')

for z in range(num_zones):
    mask = facilities['zone'] == z
    ax.scatter(facilities.loc[mask, 'x'], facilities.loc[mask, 'y'],
               marker='s', s=120, c=zone_palette[z], edgecolors='black',
               linewidths=1.5, zorder=4, label=f'Facilities (zone {z})')

for i, row in facilities.iterrows():
    if i in (0, 1, 2):
        ax.annotate(f'i={i}', (row['x'] + 1.5, row['y'] - 3), fontsize=14, color='black')

ax.set_title(f'Instance Layout  (type {ITYPE})', fontsize=13, fontweight='bold')
ax.set_xlim(-2, grid_size + 2); ax.set_ylim(-2, grid_size + 2)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3); ax.set_aspect('equal')

# --- Panel 2: DDU effect on a single customer ---
ax     = axes[1]
j_show = 0
k_vals = np.arange(max_demand + 1)
line_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

d_none   = get_customer_distributions(x_none)
ed_none_j = expected_demand(d_none, j_show)
ax.plot(k_vals, d_none[j_show], color='#999999', linewidth=2, marker='o', markersize=4,
        label=f'No facilities (mean={ed_none_j:.2f})')

fac_per_zone = {z: facilities[facilities['zone'] == z].index[0] for z in range(num_zones)}
for z in range(num_zones):
    x_single = np.zeros(num_facilities, dtype=int)
    x_single[fac_per_zone[z]] = 1
    d_single  = get_customer_distributions(x_single)
    ed_single = expected_demand(d_single, j_show)
    ax.plot(k_vals, d_single[j_show], color=zone_palette[z],
            linestyle=line_styles[z], linewidth=1.5, marker='o', markersize=4,
            label=f'Fac {fac_per_zone[z]} zone {z} (mean={ed_single:.2f})')

d_all   = get_customer_distributions(x_all)
ed_all_j = expected_demand(d_all, j_show)
ax.plot(k_vals, d_all[j_show], color='black', linewidth=2, marker='o', markersize=4,
        label=f'All facilities (mean={ed_all_j:.2f})')

ax.set_title(f'DDU Effect: Customer {j_show}  (type {ITYPE})', fontsize=13, fontweight='bold')
ax.set_xlabel('Demand realization'); ax.set_ylabel('Probability')
ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.3)

# --- Panel 3: Joint total expected demand per activation pattern ---
ax = axes[2]
joint_means   = []
labels_short  = []

for label, active_zones in region_labels:
    x_pat = np.zeros(num_facilities, dtype=int)
    for z in active_zones:
        for i in facilities[facilities['zone'] == z].index:
            x_pat[i] = 1
            break
    dists = get_customer_distributions(x_pat)
    joint_means.append(sum(expected_demand(dists, j) for j in range(num_customers)))
    labels_short.append(label)

x_pos      = np.arange(len(joint_means))
bar_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(joint_means)))
ax.bar(x_pos, joint_means, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=8)
ax.set_title(f'Total Expected Demand by Zone Subset  (type {ITYPE})',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Active zone subset'); ax.set_ylabel('Total expected demand')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
out_plot = f'instance_overview_small_{ITYPE}.png'
plt.savefig(out_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to {out_plot}")

# ===================================================================
# 11. Export instance data
# ===================================================================
instance = {
    'interaction_type': ITYPE,
    'num_customers':    num_customers,
    'num_facilities':   num_facilities,
    'num_zones':        num_zones,
    'max_demand':       max_demand,
    'R': R, 'C': C, 'O': O, 'k': k_budget, 'T': T, 'c': c_transport,
    'alpha_base': ALPHA_BASE,
    'beta_base':  BETA_BASE,
    'base_scale': BASE_SCALE,
    'customer_coords':      cust_coords.tolist(),
    'customer_base_alpha':  alpha_vals.tolist(),
    'customer_base_beta':   beta_vals.tolist(),
    'facility_coords':      facility_coords.tolist(),
    'facility_zones':       facilities['zone'].tolist(),
    'profit_matrix':        profit_matrix.tolist(),
    'dist_matrix':          dist_matrix.tolist(),
    'zone_order_per_customer': {str(j): v for j, v in zone_order_per_customer.items()},
    'zone_dists_per_customer': {
        str(j): {str(z): d for z, d in dists.items()}
        for j, dists in zone_dists_per_customer.items()
    },
    'activation_regions': [list(r) for r in activation_regions],
}

out_json = f'instance_data_small_{ITYPE}.json'
with open(out_json, 'w') as f:
    json.dump(instance, f, indent=2)
print(f"Instance data saved to {out_json}")

# ===================================================================
# 12. Summary
# ===================================================================
print("\n" + "=" * 60)
print("INSTANCE SUMMARY")
print("=" * 60)
print(f"Interaction type : {ITYPE}")
print(f"Customers        : {num_customers}")
print(f"Facilities       : {num_facilities}")
print(f"Zones            : {num_zones}")
print(f"Regions          : {len(activation_regions)} (powerset of {num_zones} zones)")
print(f"Max demand       : {max_demand}")
print(f"Revenue/unit     : {R}")
print(f"Capacity/fac     : {C}")
print(f"Opening cost     : {O}")
print(f"Budget/stage     : {k_budget}")
print(f"Stages           : {T}")
print(f"alpha_base       : {ALPHA_BASE}  (mean effect at rank n = {ALPHA_BASE}^n)")
print(f"beta_base        : {BETA_BASE}   (scale effect at rank n = {BETA_BASE}^n)")
print(f"\nFacilities per zone:")
for z in range(num_zones):
    facs = list(facilities[facilities['zone'] == z].index)
    print(f"  Zone {z}: facilities {facs}")
print(f"\nProfit matrix range: [{profit_matrix.min():.0f}, {profit_matrix.max():.0f}]")
