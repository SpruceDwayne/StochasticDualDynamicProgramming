# From Instance to Experiment: A Complete Walkthrough

This document tells the end-to-end story of the facility location experiment with decision-dependent uncertainty (DDU): how a problem instance is constructed in Python, what gets serialised to disk, and how the Julia experiment code consumes it to benchmark two stochastic optimisation algorithms.

---

## 1. The Problem in One Paragraph

A firm opens facilities over `T = 4` sequential stages. There are `I = 8` candidate sites grouped into `Z = 5` geographic zones, and `J = 15` customers scattered across a 200 × 200 grid. At each stage the firm can open at most `k` new facilities (permanently), then earns revenue by assigning open facilities to serve realised customer demand. The twist: **which facilities are open determines the demand distribution for the next stage**. Opening facilities in a zone signals market presence and shifts customer demand upward (or, in some configurations, downward for distant zones). This feedback loop between investment decisions and future demand is the DDU structure.

---

## 2. Instance Generation (`generate_instance.py`)

Run once per interaction type:

```bash
python generate_instance.py --interaction A   # or B, C, D
```

The script constructs all structural data from a fixed random seed (`np.random.seed(420)`) so results are reproducible, and writes two artefacts: a PNG overview plot and a JSON instance file.

### 2.1 Spatial layout

- **Customers** are placed uniformly at random on a 200 × 200 grid. Base demand parameters `(α_j, β_j)` are derived from linearly spaced target means (1.5 to 5) via a BetaBinomial moment-matching formula.
- **Facilities** are placed in two groups: three are deliberately located near the three highest-demand customers (within ±10 units), and five are scattered in the centre of the grid. This ensures some facilities are commercially attractive from the start.
- **Zones** are obtained by K-means clustering (`k = 5`) of the facility coordinates. Each facility belongs to exactly one zone.

### 2.2 Economics

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Revenue per unit served | `R` | 500 |
| Facility capacity | `C` | 25 |
| Fixed opening cost | `O` | 1500 |
| Max new openings per stage | `k` | K_max |
| Stages | `T` | 4 |
| Transport cost coefficient | `c` | 2 |

The **profit matrix** `p[i,j] = R − c · dist(i,j) · 0.002` converts Euclidean distance into a per-unit margin. Closer facilities are more profitable to serve from.

### 2.3 The DDU mechanism

The 5 zones admit `2^5 = 32` possible activation patterns (the powerset). Each pattern is a **region**. When a set of facilities is open, the active zone set is determined, which selects one of the 32 regions. That region parameterises the BetaBinomial demand distributions for the next stage.

For each customer `j` and active zone vector, the demand distribution is:

```
mean_j  = base_mean_j  × (1 + mean_delta_j)
scale_j = BASE_SCALE   × (1 + scale_delta_j)   [BASE_SCALE = 6]
ξ_j ~ BetaBinomial(n=10, α=mean_j/n × scale_j, β=(1−mean_j/n) × scale_j)
```

How `mean_delta` and `scale_delta` are computed defines the **interaction type**.

### 2.4 The four interaction types

All types use the same decay bases: `ALPHA_BASE = 0.7` (mean effect) and `BETA_BASE = 0.6` (scale effect). For each customer `j`, zones are ranked by distance to their centroid, yielding `zone_order[j]`.

**Type A — All zones, additive, by rank**

Every active zone contributes to mean and scale, with the contribution decaying geometrically with proximity rank:

```
mean_delta  += 0.7^rank × active[zone_order[j][rank]]
scale_delta += 0.6^rank × active[zone_order[j][rank]]
```

More zones open → higher and more dispersed demand for all customers. Strongest incentive to open many facilities early.

**Type B — Nearest zone only**

Only whether the single nearest zone is active matters:

```
mean_delta  = 0.7 × active[zone_order[j][0]]
scale_delta = 0.6 × active[zone_order[j][0]]
```

Binary effect per customer. Simpler DDU structure; the decision of *which* zone to activate first dominates.

**Type C — Nearest active zone**

The effect comes from the nearest zone that is currently active, with magnitude depending on how far away that zone is:

```
for rank = 1, 2, ..., 5:
    if zone_order[j][rank] is active:
        mean_delta  = 0.7^rank
        scale_delta = 0.6^rank
        break
```

Cascading logic: having a close active zone is better than a distant one. Zone ordering matters.

**Type D — Mixed signs, alternating by rank**

The nearest zone is beneficial; more distant zones are detrimental:

```
sign = +1 (rank 1)  or  −1 (rank ≥ 2)
mean_delta  += 0.7^rank × sign × active[zone_order[j][rank]]
scale_delta += 0.6^rank × sign × active[zone_order[j][rank]]
```

Non-monotone structure: opening too many zones can reduce demand for some customers. 

### 2.5 Output: the instance JSON

After generation, the script writes `instance_data_{ITYPE}.json` containing every quantity needed by the solver: coordinates, profit matrix, distance matrix, zone assignments, zone-order-per-customer, and the full list of 32 activation regions. Crucially, the JSON does **not** pre-compute the 32 demand distributions — those are reconstructed on the fly by the Julia code using the stored `alpha_base`, `beta_base`, `base_scale`, and per-customer `(α_j, β_j)` pairs together with the interaction type.

---

## 3. The Experiment (`EXPERIMENT.md` / `run_comparison_G.jl`)

### 3.1 What is being compared

Two algorithms are trained on the same instance and then evaluated on the same 1000 out-of-sample paths:

| Algorithm | Demand model during training |
|-----------|------------------------------|
| **DDU-SDDiP** | Region-conditional; the sampled distribution shifts with the current facility state |
| **Standard SDDiP** | Exogenous; always samples from region 1 (no facilities open) regardless of state |

The central question is the **value of modelling DDU**: how much profit is left on the table by ignoring the feedback between facility decisions and demand?

### 3.2 SAA discretisation

The continuous demand support `{0,…,10}^15` is intractable. Before training, both algorithms fix a **Sample Average Approximation**:

- For each of the 32 regions, draw `N` demand vectors independently from the BetaBinomial distributions associated with that region (seed 123).
- Scenarios are held fixed for all training iterations.
- Experiments are run with `N ∈ {25, 50}`.

DDU-SDDiP uses all 32 × N scenario sets. Standard SDDiP uses only the N scenarios from region 1 (equivalent to no facilities opening).

### 3.3 Training

**DDU-SDDiP forward pass:** at each stage, the sampler draws from the scenario set of the *current region*, which is determined by the facilities opened so far. The region propagates through the trajectory.

**DDU-SDDiP backward pass:** builds region-specific Lagrangian cuts of the form:
```
θ ≥ α^{d,k} + (β^{d,k})ᵀ x_t − M_big × (1 − δ_d)
```
The big-M term deactivates cuts for the wrong region. Cuts for different regions coexist in the same value-function approximation without interfering.

**Standard SDDiP forward pass:** always samples from region 1, so the policy is trained as if demand never responds to facility openings.

**Standard SDDiP backward pass:** cuts are also built against region-1 scenarios only. The resulting value function is optimal for the wrong (exogenous) problem and is then applied to the true DDU dynamics in evaluation.

Both algorithms use Lagrangian cuts throughout (`cut_type = :lagrangian`) with the level method for the Lagrangian dual, and run for at most 150 iterations with early stopping (`patience = 10`).

### 3.4 Out-of-sample evaluation

After training, both policies are evaluated on **1000 independent sample paths** generated from the true DDU demand distribution (seed 1111). At each stage:

1. Identify the true region from the current facility state.
2. Draw a demand vector from the correct conditional distribution.
3. Solve the stage subproblem with realized demand and the trained cuts.
4. Record stage profit (revenue minus opening cost, excluding the cut approximation `θ`).
5. Advance the facility state.

Both policies face identical scenario seeds, so the comparison is fair.

### 3.5 Experimental grid

```
run_all.sh
```

runs all 8 combinations:

| Instance type | N_SAA | DDU structure |
|---------------|-------|---------------|
| A | 25, 50 | All zones additive — strongest DDU incentive |
| B | 25, 50 | Nearest zone only — binary, simpler |
| C | 25, 50 | Nearest active zone — cascading |
| D | 25, 50 | Mixed signs — non-monotone, hardest to exploit |

---

## 4. How the Pieces Connect

```
generate_instance.py --interaction {A,B,C,D}
         │
         │  writes
         ▼
instance_data_{ITYPE}.json
  ├── coordinates, zones, profit_matrix, dist_matrix
  ├── customer_base_alpha / beta  (BetaBinomial base parameters)
  ├── zone_order_per_customer     (proximity ranking per customer)
  ├── activation_regions          (all 32 zone subsets)
  └── alpha_base, beta_base, base_scale, interaction_type
         │
         │  loaded by
         ▼
run_comparison_G.jl
  ├── reconstructs 32 BetaBinomial demand distributions (one per region)
  ├── draws N SAA scenarios per region (fixed seed)
  ├── trains DDU-SDDiP  (region-aware cuts, all 32 scenario sets)
  ├── trains Standard SDDiP  (region-1 scenarios only, no DDU structure)
  └── evaluates both on 1000 true-DDU out-of-sample paths
         │
         │  writes to results/{ITYPE}/
         ▼
  comparison_summary_G_{ITYPE}_N{N}.csv   (mean profit, std, LB, timing)
  convergence_data_G_{ITYPE}_N{N}.csv     (per-iteration LB)
  convergence_plot_G_{ITYPE}_N{N}.png     (convergence curves)
```

The JSON is the contract between the Python instance generator and the Julia solver. Everything the solver needs to reconstruct distributions and evaluate profits is in that file; no other communication between the two languages is required.

---

## 5. What the Results Tell Us

The gap `VOI_DDU = mean_profit(DDU-SDDiP) − mean_profit(Standard SDDiP)` measures how much value the DDU structure contributes. A large positive gap means the DDU-aware policy successfully exploits the feedback loop — opening the right facilities early to shift demand upward in subsequent stages.

### K_max = 1 — one new facility per stage

With only one opening allowed per stage, both policies achieve similar out-of-sample profit. DDU-SDDiP provides no advantage and is slightly below Standard SDDiP on types C and D.

| Type | N_SAA | DDU Avg Profit | DDU Std | STD Avg Profit | STD Std | Gap (%) |
|------|-------|---------------|---------|---------------|---------|---------|
| A | 25 | 118,587 | 1,108 | 118,528 | 1,063 | +0.05 |
| A | 50 | 118,587 | 1,108 | 118,566 | 991 | +0.02 |
| B | 25 | 103,614 | 5,565 | 103,596 | 5,727 | +0.02 |
| B | 50 | 103,614 | 5,565 | 104,551 | 4,942 | −0.90 |
| C | 25 | 108,291 | 3,837 | 108,979 | 3,981 | −0.63 |
| C | 50 | 108,291 | 3,837 | 109,115 | 3,852 | −0.75 |
| D | 25 | 76,872 | 8,866 | 79,066 | 7,541 | −2.77 |
| D | 50 | 76,872 | 8,866 | 81,185 | 6,834 | −5.31 |

Most runs hit the 200-iteration limit. The DDU lower bounds are substantially more negative than the Standard SDDiP bounds (e.g. −133,000 vs. −71,900 for type A, N=25), reflecting the conservatism introduced by the big-M cuts.

### K_max = 2 — two new facilities per stage

With a wider per-stage budget the DDU advantage begins to emerge, though gains are modest and vary with N_SAA. Type D continues to show mixed results; types A and B are positive with N=50 but not always with N=25.

| Type | N_SAA | DDU Avg Profit | DDU Std | STD Avg Profit | STD Std | Gap (%) |
|------|-------|---------------|---------|---------------|---------|---------|
| A | 25 | 178,208 | 7,077 | 177,820 | 8,041 | +0.22 |
| A | 50 | 178,208 | 7,077 | 177,678 | 10,287 | +0.30 |
| B | 25 | 127,812 | 9,243 | 130,393 | 8,730 | −1.98 |
| B | 50 | 127,812 | 9,243 | 123,970 | 8,807 | +3.10 |
| C | 25 | 137,069 | 7,622 | 137,645 | 8,015 | −0.42 |
| C | 50 | 137,069 | 7,622 | 135,616 | 8,004 | +1.07 |
| D | 25 | 88,665 | 8,249 | 94,283 | 7,238 | −5.96 |
| D | 50 | 88,665 | 8,249 | 87,426 | 7,204 | +1.42 |

Several DDU runs converge before the iteration limit (e.g. type A at 78–98 iterations, type B at 106–140), in contrast to K_max = 1 where almost all runs exhaust 200 iterations.

### K_max = 3 — three new facilities per stage

The DDU advantage is large and consistent across all interaction types and both values of N_SAA. The DDU-aware policy substantially outperforms the exogenous baseline.

| Type | N_SAA | DDU Avg Profit | DDU Std | STD Avg Profit | STD Std | Gap (%) |
|------|-------|---------------|---------|---------------|---------|---------|
| A | 25 | 182,498 | 9,960 | 165,146 | 16,698 | +10.51 |
| A | 50 | 182,498 | 9,960 | 164,189 | 13,178 | +11.15 |
| B | 25 | 132,060 | 10,864 | 107,447 | 9,364 | +22.91 |
| B | 50 | 132,060 | 10,864 | 107,914 | 7,860 | +22.38 |
| C | 25 | 139,522 | 8,606 | 128,280 | 9,522 | +8.76 |
| C | 50 | 139,522 | 8,606 | 127,922 | 8,686 | +9.07 |
| D | 25 | 91,198 | 9,582 | 68,890 | 8,116 | +32.38 |
| D | 50 | 91,198 | 9,582 | 67,436 | 6,847 | +35.24 |

Type D (mixed signs, non-monotone) shows the largest gap (32–35%), followed by type B (nearest zone only, 22%). Type A (all zones additive) and type C (nearest active zone) show moderate gains around 9–11%. The Standard SDDiP standard deviation is notably higher than DDU's in all type-A and type-B runs, reflecting that the exogenous policy makes more variable facility choices when evaluated on the true DDU dynamics. DDU runs converge early in most cases (78–150 iterations).

### Computational cost

DDU-SDDiP is substantially more expensive than Standard SDDiP across all settings. Representative wall times (N=50):

| Setting | DDU Time (s) | STD Time (s) |
|---------|-------------|-------------|
| K1, B | 34,858 | 1,233 |
| K2, C | 24,217 | 1,157 |
| K3, C | 15,330 | 1,251 |
| K3, A | 777 | 613 |

The cost difference stems from region-specific cut management (32 cut sets, each with big-M constraints) and the larger backward pass (all N scenarios per region). Standard SDDiP operates on a single scenario set and has no region bookkeeping overhead.

