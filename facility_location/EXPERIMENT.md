# Facility Location Experiment: DDU-SDDiP vs Standard SDDiP

## What we investigate

We compare two algorithmic policies on a **multi-stage stochastic facility location problem with decision-dependent uncertainty (DDU)**:

1. **DDU-SDDiP** — an algorithm that explicitly models and exploits the decision-dependent demand structure during training.
2. **Standard SDDiP** — a classical algorithm that treats the demand distribution as exogenous and fixed, ignoring the DDU structure.

The central question is: **how much does ignoring decision-dependent demand cost in terms of out-of-sample profit?**

The comparison is run across four problem variants (interaction types A–D) that differ in *how* facility openings shift demand distributions, and across two SAA sample sizes (N=25 and N=50).

---

## Problem description

### Setting

A firm opens facilities sequentially over `T = 4` stages. There are `I = 8` candidate facilities, `J = 15` customers, and `Z = 5` geographic zones. Each facility belongs to exactly one zone.

At each stage the firm:
- Chooses which additional facilities to open (at most `k = K_max` new openings per stage).
- Once a facility is opened it stays open (monotone state dynamics).
- Earns revenue by assigning open facilities to serve realized customer demand.
- Pays a fixed opening cost `O = 1500` per newly opened facility.

The state entering stage `t` is `x_{t-1} ∈ {0,1}^8`, indicating which facilities are currently open. The initial state is `x_0 = 0`.

### Stage-t subproblem

**Decision variables:**

| Variable | Type | Description |
|----------|------|-------------|
| `x_t ∈ {0,1}^8` | Binary | Facility configuration after stage t |
| `w_{ij} ≥ 0` | Continuous | Demand of customer j served by facility i |
| `a_z ∈ {0,1}` | Binary | Zone z activation indicator (DDU model only) |
| `δ_d ∈ {0,1}` | Binary | Region d indicator (DDU model only, exactly one active) |
| `θ` | Continuous | Cost-to-go epigraph variable |

**Objective (minimise negative profit):**
```
min  −Σ_{i,j} p[i,j] · w[i,j]           (revenue from serving demand)
    + O · Σ_i (x[i,t] − x[i,t−1])       (opening cost for newly opened facilities)
    + θ                                   (approximate future cost)
```

**Constraints:**
```
Σ_i w[i,j]  ≤ ξ[j]          ∀ j    (serve at most realized demand)
Σ_j w[i,j]  ≤ C · x[i,t]   ∀ i    (facility capacity C = 25)
x[i,t]      ≥ x[i,t−1]     ∀ i    (monotonicity)
Σ_i (x[i,t] − x[i,t−1]) ≤ k       (budget: at most k = 3 new openings)
```

The profit matrix is `p[i,j] = R − c · dist(i,j)` with base revenue `R = 500` and transportation cost `c` per unit distance, computed from Euclidean distances loaded from the instance file.

### DDU mechanism: zones and regions

The key DDU structure is:

1. **Zone activation:** Zone `z` is *active* after stage `t` if and only if at least one facility in zone `z` is open: `a_z = 1 ⟺ Σ_{i ∈ I_z} x[i,t] ≥ 1`.

2. **Region identification:** There are `2^5 = 32` possible activation patterns (one for each subset of the 5 zones). Each such subset is a *region*. The unique region `d` active after stage `t` is the one whose zone-activation set exactly matches `{z : a_z = 1}`.

3. **Decision-dependent demand:** Next-stage demand for each customer `j` is drawn from a BetaBinomial distribution whose parameters depend on which region is active. Opening more facilities (activating more zones) generally shifts the demand distribution — but the direction and magnitude of the shift varies by interaction type.

In the DDU subproblem (DDU-SDDiP), auxiliary binary variables `a` and `δ` encode zone activation and region selection, and are linked to `x_t` via constraints. Exactly one `δ_d` equals 1 per stage, selecting the active conditional distribution for the next stage.

### Demand distribution

For each customer `j`, demand `ξ_j ~ BetaBinomial(n, α_j·s, β_j·s)` where:
- `n = 10` is the maximum demand.
- `α_j, β_j` are customer-specific base shape parameters.
- `s` (base scale = 6) is a dispersion parameter.
- Both the mean and scale are shifted by a region-specific delta computed from the active zones and interaction type (see below).

---

## Instance types: the four interaction patterns

All four instances share identical structural parameters (dimensions, costs, coordinates). They differ only in how zone activations shift demand parameters, controlled by `interaction_type` and `(alpha_base, beta_base) = (0.7, 0.6)`:

### Type A — All zones, additive, by rank

Every active zone contributes additively to both mean and scale. The contribution decreases geometrically with the zone's distance rank for customer `j`:

```
mean_delta  += alpha_base^rank  ·  active[zone_order[j][rank]]
scale_delta += beta_base^rank   ·  active[zone_order[j][rank]]
```

All active zones shift demand upward, with the nearest zone having the largest effect. The more zones active, the higher and more dispersed demand becomes. This creates the strongest incentive to open facilities.

### Type B — Nearest zone only

Only the single nearest zone to each customer matters:

```
mean_delta  = alpha_base  if zone_order[j][1] is active  else 0
scale_delta = beta_base   if zone_order[j][1] is active  else 0
```

The demand effect is binary (the nearest zone is either active or not) and independent across customers beyond the first zone. Simpler DDU structure than A.

### Type C — Nearest active zone

The contribution comes from the *nearest active zone* in the ranked ordering for each customer:

```
for rank = 1, 2, ..., num_zones:
    if zone_order[j][rank] is active:
        mean_delta  = alpha_base^rank
        scale_delta = beta_base^rank
        break
```

The effect diminishes the farther away the nearest active zone is. This is a "cascading" structure: the first active zone in proximity order wins.

### Type D — Mixed signs, alternating by rank

Zones alternate between positive and negative effects:

```
for rank = 1, 2, ..., num_zones:
    sign = +1 if rank == 1  else  −1
    mean_delta  += alpha_base^rank · sign · active[zone_order[j][rank]]
    scale_delta += beta_base^rank  · sign · active[zone_order[j][rank]]
```

The nearest zone increases demand while more distant zones decrease it. This creates a non-monotone relationship between zone coverage and demand, making the DDU structure harder to exploit.

---

## SAA discretization

The full demand support `{0,...,10}^15` is intractable. We fix a **Sample Average Approximation** before training:

- For each of the 32 regions, draw `N` independent demand vectors (one per customer, independently from their BetaBinomial) using a fixed seed (123).
- All N scenarios get uniform weight `1/N`.
- These scenario sets are held fixed throughout all iterations of both algorithms.

The experiments are run with `N ∈ {25, 50}`.

---

## The two algorithms

### DDU-SDDiP

Implements the full DDU-aware algorithm from `run_comparison_G.jl` using `DDUSDDP` / `run_ddu_sddip!`.

**Forward pass:** At each stage `t`, the sampler draws a demand vector from the scenario set associated with the *current region*:
```julia
(ζ) -> ddu_regions[ζ].Xi[rand(1:N_SAA)]   # ζ = current region index
```
The region `ζ` is propagated via `noop_ctx` (identity): the region chosen at stage `t` determines which distribution is sampled at stage `t+1`.

**Stage-1 demand:** Deterministic expected demand under region 1 (no facilities open), since `x_0 = 0` implies region 1 is always active at stage 1.

**Backward pass:** Builds region-specific Lagrangian cuts. For each visited state `x_t` and active region `d`, subproblems for all N scenarios under region `d` are solved and their results averaged to form a cut:
```
θ ≥ α^{d,k} + (β^{d,k})ᵀ x_t − M_big · (1 − δ_d)
```
The big-M term deactivates the cut when region `d` is not selected. Cuts for different regions do not interfere.

**Lower bound:** The stage-1 objective value after each iteration, which is a valid lower bound on the true DDU optimal value.

**Cut type:** Lagrangian cuts throughout (`cut_type = :lagrangian`, `burnin_iters = 0`). The level method is used to solve the Lagrangian dual (configured via `LevelMethodConfig`).

**Hyperparameters:** `max_iter = 150`, `patience = 10`, `force_every = 2`, `cut_atol = 1e-6`, `S = 1` forward trajectory per iteration.

### Standard SDDiP (ignoring DDU)

Implements a classical exogenous-uncertainty SDDiP using `SDDP` / `run_sddip!`.

**Critical difference — forward pass:** All stages always sample from `saa[1]` (region 1, the baseline distribution for no facilities open), regardless of which facilities were chosen:
```julia
() -> saa[1][rand(rng_std, 1:N_SAA), :]   # always region 1
```
The sampler takes no argument and cannot observe `x_t`.

**Critical difference — backward pass:** The `children` function also always returns the region-1 scenario set:
```julia
(t, ctx) -> (Any[saa[1][n, :] for n in 1:N_SAA], uniform_p)
```
So cuts are built against demand scenarios drawn from the distribution as if no facilities were ever open, regardless of the actual state.

**Subproblem:** `make_exogenous_stage_builder` — no zone activation variables (`a`) or region indicator variables (`δ`). Demand `ω` is a fixed parameter passed directly into the service constraints.

**Lower bound:** Recomputed after each iteration by solving the stage-1 model with `ω = expected_d1` (deterministic expected demand under region 1) and the current cut pool.

**What this algorithm solves:** Effectively a different problem — one where demand is permanently distributed as if no facilities were ever open. The policy it produces is optimal for that simpler (incorrect) problem, but is applied to the true DDU problem in evaluation.

---

## Out-of-sample evaluation

After training, both policies are evaluated on 1000 independent sample paths generated from the **true DDU demand distribution**:

```julia
x_int  = round(x_prev)
d_true = identify_region(inst, x_int)          # true region based on actual x
ξ      = [rand(rng, _customer_dist(inst, j, region_to_zone_active(inst, d_true)))
          for j in 1:num_customers]
```

At each stage the demand is sampled from the correct conditional distribution given the facilities opened so far. Both policies face identical scenario seeds (seed = 1111).

For each path and stage:
- Solve the stage subproblem with the realized `ξ` and incoming state `x_prev`.
- Record stage profit = revenue − opening cost (i.e. objective excluding `θ`).
- Advance `x_prev` using the rounded next-stage facility vector.

The reported metrics are **mean profit** and **standard deviation of profit** across the 1000 paths.

This evaluation is fair: both policies are penalised/rewarded by the same true DDU dynamics, regardless of what each assumed during training.

---

## Experimental grid

`run_all.sh` runs every combination of instance type and SAA sample size:

| Instance type | N_SAA | Description |
|---------------|-------|-------------|
| A | 25 | All zones, additive, 25 scenarios |
| A | 50 | All zones, additive, 50 scenarios |
| B | 25 | Nearest zone only, 25 scenarios |
| B | 50 | Nearest zone only, 50 scenarios |
| C | 25 | Nearest active zone, 25 scenarios |
| C | 50 | Nearest active zone, 50 scenarios |
| D | 25 | Mixed signs, 25 scenarios |
| D | 50 | Mixed signs, 50 scenarios |

Total: **8 runs**, each producing results for both DDU-SDDiP and Standard SDDiP.

### How to run

```bash
# All 8 combinations (logs to facility_location/results/<TYPE>/run_G_<TYPE>_N<N>.log)
bash run_all.sh

# Single instance type, both N values
bash run_all.sh A

# Single run
bash run_all.sh A 25

# Single run directly
julia problem_src/run_comparison_G.jl A 25
```

Failed runs are recorded in `results/failed_runs.txt`.

---

## Output files

Each run writes three files to `results/<TYPE>/`:

| File | Content |
|------|---------|
| `comparison_summary_G_<TYPE>_N<N>.csv` | One row per policy: avg profit, std dev, first-stage facilities opened, active region, final LB, iterations, wall time |
| `convergence_data_G_<TYPE>_N<N>.csv` | Per-iteration lower bound and wall time for both policies |
| `convergence_plot_G_<TYPE>_N<N>.png` | Side-by-side plots: iteration vs LB and wall time vs LB for both policies |

---

## Interpreting results

The **value of modelling DDU** is the gap in mean out-of-sample profit:
```
VOI_DDU = mean_profit(DDU-SDDiP) − mean_profit(Standard SDDiP)
```

A positive gap indicates that the DDU-aware policy finds better facility sequences — ones that open facilities which shift demand distributions in a more profitable direction, and are timed appropriately given the `k = 3` per-stage budget.

The standard policy's lower bound is a lower bound for a *wrong* problem (exogenous region-1 demand throughout). It will generally be different compared to true out-of-sample performance when the DDU effect is strong, because it assumes the same (lowest-demand) distribution regardless of which facilities are opened.

The interaction type affects how large the VOI_DDU is expected to be:
- **Type A** creates the strongest incentive to open multiple facilities early, since each additional active zone adds more demand. The DDU effect compounds.
- **Type B** has a binary, all-or-nothing effect per customer. The value of sequencing is present but simpler to exploit.
- **Type C** is similar to B but the value of being the *nearest* active zone makes zone ordering matter more.
- **Type D** has a non-monotone structure — opening too many zones can reduce demand for some customers. This makes the sequencing problem qualitatively different.


We run experiment for K_max = 1,2,3 and get results: