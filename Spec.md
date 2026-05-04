# Facility Location with Decision-Dependent Uncertainty — Implementation Spec

## What this document is

This is a specification for implementing a **multistage facility location problem with decision-dependent uncertainty (DDU)** in Julia. The problem will be solved using an existing DDU-SDDP/SDDiP solver. The task is to build the **problem instance**, **stage subproblems**, and **DDU mapping** so they can be passed to the solver.

---

## Problem overview

A firm decides which facilities to open over T=4 stages. Opening facilities affects future customer demand (decision-dependent uncertainty). The firm earns revenue by serving demand, pays fixed costs to open new facilities, and wants to maximize total expected profit.

The key DDU mechanism: facilities are grouped into **zones**. Which zones are "active" (have ≥1 open facility) determines the probability distribution of next-stage demand. A per-stage budget constraint limits the firm to opening at most k=2 new facilities per stage, creating a nontrivial sequencing problem: the firm must decide which facilities to open first, knowing that early openings shift demand distributions for all subsequent stages.

---

## Sets and indices

| Symbol | Description | Size |
|--------|-------------|------|
| I | Facilities | 8 (indexed 1:8) |
| J | Customers | 15 (indexed 1:15) |
| Z | Zones | 3 (indexed 1:3) |
| D | Regions (activation patterns) | 8 (powerset of Z, indexed 1:8) |
| T | Stages | 4 |

### Facility-to-zone mapping

| Zone | Facilities (1-indexed) |
|------|----------------------|
| 1 | 1, 2, 6, 7, 8 |
| 2 | 4, 5 |
| 3 | 3 |

### Region definitions (powerset of zones)

| Region d | Active zones |
|----------|-------------|
| 1 | ∅ (none) |
| 2 | {1} |
| 3 | {2} |
| 4 | {3} |
| 5 | {1, 2} |
| 6 | {1, 3} |
| 7 | {2, 3} |
| 8 | {1, 2, 3} |

---

## Parameters (all constant across stages)

| Parameter | Value | Description |
|-----------|-------|-------------|
| R | 400 | Base revenue per unit demand served |
| C | 15 | Capacity per facility (uniform) |
| O | 500 | Fixed cost of opening a facility (uniform) |
| c | 2 | Transportation cost per unit Euclidean distance |
| k | 2 | Maximum number of new facility openings per stage |
| n | 10 | Maximum demand per customer (BetaBinomial support: {0,...,10}) |
| s | 6 | BetaBinomial scale parameter |
| T | 4 | Number of stages |

### Profit matrix

```
p[i,j] = R - c * dist(i,j)    for all (i,j) ∈ I × J
```

where `dist(i,j)` is the Euclidean distance between facility i and customer j. All entries are positive (range ≈ 183 to 394). The full profit matrix and coordinates should be loaded from `instance_data.json`.

---

## State variables

The state entering stage t is:

- **x_{t-1} ∈ {0,1}^8**: which facilities are currently open

Total state dimension: 8 binary variables.

Initial state: `x_0 = 0` (no facilities open).

Since facilities once opened stay open (`x[i,t] ≥ x[i,t-1]`), the state is monotonically nondecreasing. The zone activation pattern — and hence the active region — is fully determined by `x_t`.

---

## Stage-t subproblem

### Decision variables

| Variable | Type | Dimension | Description |
|----------|------|-----------|-------------|
| x_t | Binary | 8 | Facility configuration after stage t |
| w_{ij} | Continuous ≥ 0 | 8 × 15 | Demand from customer j served by facility i |
| a_z | Binary | 3 | Zone activation indicator (auxiliary) |
| δ_d | Binary | 8 | Region indicator |
| θ | Continuous | 1 | Cost-to-go approximation variable |

### Objective (minimize)

```
min  - Σ_{i,j} p[i,j] * w[i,j]          (negative profit from serving demand)
     + Σ_i O * (x[i,t] - x[i,t-1])      (opening cost for newly opened facilities)
     + θ                                  (approximate future cost)
```

Note: since `x[i,t] ≥ x[i,t-1]` is enforced, `(x[i,t] - x[i,t-1])` is already nonneg.

### Constraints

**Demand service:**
```
Σ_i w[i,j] ≤ ξ[j,t]              ∀ j ∈ J     (serve at most realized demand)
Σ_j w[i,j] ≤ C * x[i,t]          ∀ i ∈ I     (capacity, 0 if closed)
w[i,j] ≥ 0                        ∀ i,j
```

**Facility monotonicity (once open, stays open):**
```
x[i,t] ≥ x[i,t-1]                 ∀ i ∈ I
```

**Per-stage opening budget:**
```
Σ_i (x[i,t] - x[i,t-1]) ≤ k      (at most k=2 new openings per stage)
```

**Zone activation:**
```
a[z] ≤ Σ_{i ∈ I_z} x[i,t]         ∀ z ∈ Z     (if no facility open, zone inactive)
a[z] ≥ x[i,t]                      ∀ i ∈ I_z, z ∈ Z  (if any facility open, zone active)
a[z] ∈ {0,1}
```

Equivalently: `a[z] = 1 iff Σ_{i ∈ I_z} x[i,t] ≥ 1`.

**Region activation:**

Each region d corresponds to a specific subset S_d ⊆ Z of active zones. The region indicator δ is linked to a:

```
Σ_d δ[d] = 1                                    (exactly one region active)
δ[d] ∈ {0,1}                                    ∀ d ∈ D

# Linking δ to a:
a[z] = Σ_{d : z ∈ S_d} δ[d]                     ∀ z ∈ Z
```

This single set of equalities, combined with `Σ_d δ[d] = 1`, fully links δ to a.

**Region-activated cuts (from DDU-SDDP backward pass):**
```
θ ≥ v^{d,k} + (β^{d,k})ᵀ x_t - M_d * (1 - δ[d])    ∀ d, k
```

where `v^{d,k}`, `β^{d,k}` are cut coefficients from iteration k for region d, and `M_d` is a big-M constant.

### Terminal stage

At stage T, set `θ = 0` (no future cost). The subproblem is just the demand-serving LP plus opening costs.

---

## DDU mapping: how to compute demand distributions

Given a facility configuration `x_t` at stage t, the next-stage demand distribution is fully determined by the zone activation pattern `a(x_t)`.

### Step 1: Identify the active region

Compute `a[z] = 1{Σ_{i ∈ I_z} x[i,t] ≥ 1}` for each zone. The region d is the unique index such that `S_d = {z : a[z] = 1}`.

### Step 2: Compute demand distribution per customer

For each customer j:

1. Look up `zone_order[j]` — the 3 zones sorted by distance from customer j to zone centroid (loaded from `instance_data.json`, field `zone_order_per_customer`). **Note:** the JSON keys are 0-indexed strings; convert to 1-indexed.

2. Compute effective mean:
   ```
   base_mean[j] = 10 * α[j] / (α[j] + β[j])
   mean[j] = base_mean[j]
   for k = 1, 2, 3:
       z = zone_order[j][k]
       if a[z] == 1:
           mean[j] += (0.5^k) * base_mean[j]
   mean[j] = min(mean[j], 10 - 1e-6)
   ```

3. Compute BetaBinomial parameters:
   ```
   p = mean[j] / 10
   â = p * 6
   b̂ = (1 - p) * 6
   ```

4. The demand `ξ[j]` is drawn from `BetaBinomial(10, â, b̂)` with support `{0, 1, ..., 10}`.

### SAA discretization

The full joint support of customer demands has 11^15 elements per region, which is intractable. We use a **Sample Average Approximation (SAA)**: before running the algorithm, draw a fixed set of N scenarios for each region and treat them as the true finite support throughout all iterations.

**Setup (done once before the algorithm starts):**
For each region d ∈ {1,...,8}:
1. Compute the per-customer BetaBinomial PMFs for region d (as above).
2. Draw N i.i.d. demand vectors ξ^{d,1}, ..., ξ^{d,N} ∈ {0,...,10}^15 by sampling each customer independently from their PMF.
3. Assign equal probability 1/N to each scenario.

Store these as a 3D array: `scenarios[d][n, j]` = demand of customer j in scenario n of region d. Shape: 8 regions × N scenarios × 15 customers.

**Forward pass:** At each stage, once the active region d is identified, sample uniformly from {ξ^{d,1}, ..., ξ^{d,N}} to obtain the next-stage realization.

**Backward pass:** To compute the cut coefficients for region d, solve the stage-(t+1) subproblem for every scenario ξ^{d,1}, ..., ξ^{d,N} and average the results. This expectation is exact over the SAA support.

**Key property:** The scenario sets are fixed across all iterations. This ensures that the lower bound is monotonically nondecreasing and the finite convergence result (Proposition 5.2) applies directly.

A reasonable starting value is N = 50 scenarios per region.

---

## Data loading

All instance data is in `instance_data.json`. Key fields:

| JSON field | Julia type | Description |
|------------|-----------|-------------|
| `num_customers` | Int | 15 |
| `num_facilities` | Int | 8 |
| `num_zones` | Int | 3 |
| `max_demand` | Int | 10 |
| `R`, `C`, `O` | Int | 400, 15, 500 |
| `k` | Int | 2 |
| `T` | Int | 4 |
| `customer_coords` | Matrix 15×2 | Customer (x,y) positions |
| `customer_base_alpha` | Vector 15 | α_j parameters |
| `customer_base_beta` | Vector 15 | β_j parameters |
| `facility_coords` | Matrix 8×2 | Facility (x,y) positions |
| `facility_zones` | Vector 8 | Zone assignment (0-indexed in JSON → add 1) |
| `profit_matrix` | Matrix 8×15 | p[i,j] values |
| `dist_matrix` | Matrix 8×15 | Euclidean distances |
| `zone_order_per_customer` | Dict str→list | Per-customer zone ordering (0-indexed → add 1) |
| `alpha_weights` | Vector 3 | [0.5, 0.25, 0.125] |
| `activation_regions` | List of lists | Region → active zone sets (0-indexed → add 1) |

**Indexing convention:** The JSON uses 0-based indexing (from Python). Convert everything to 1-based when loading into Julia.

---

## What the solver expects (interface contract)

You need to provide the solver with:

1. **Instance data struct** containing all parameters, sets, and precomputed data (including the fixed SAA scenarios).

2. **A function that builds the stage-t subproblem** as a JuMP model, given:
   - Stage index t
   - Incoming state x_{t-1}
   - Uncertainty realization ξ_t
   - Current cut collection for each region

3. **The fixed SAA scenario sets** for each region d, used by both the forward pass (sample from) and backward pass (iterate over).

4. **Region identification function**: given a solution x_t, return which region d is active.

Consult the solver's existing interface/API and match these to whatever types and function signatures it expects.

---

## Suggested file structure

```
facility_location/
├── instance_data.json          # Generated instance (already exists)
├── src/
│   ├── FacilityLocation.jl     # Module file
│   ├── data.jl                 # Load and parse instance_data.json
│   ├── distributions.jl        # DDU mapping: region → demand distributions
│   ├── subproblem.jl           # Build JuMP stage subproblem
│   └── run.jl                  # Main script: load data, configure solver, run
└── test/
    ├── test_data.jl            # Verify data loads correctly
    ├── test_distributions.jl   # Verify DDU distributions match Python reference
    └── test_subproblem.jl      # Verify single-stage problem solves correctly
```

---

## Verification checklist

Before running the full DDU-SDDP, verify:

- [ ] Profit matrix matches Python output (range ≈ 183–394)
- [ ] Zone assignments match (zone 1: facs {1,2,6,7,8}, zone 2: {4,5}, zone 3: {3})
- [ ] For x = zeros(8) (no zones active, region 1): customer 1 mean demand = 1.5, customer 15 mean = 8.5
- [ ] For x = ones(8) (all zones active, region 8): customer 8 mean demand ≈ 9.38
- [ ] Region activation: x = [1,0,0,0,0,0,0,0] (only fac 1 open → zone 1 active) → region 2
- [ ] Single-stage subproblem solves and returns sensible profit
- [ ] Opening cost is only charged for *newly* opened facilities
- [ ] Budget constraint: with x_prev=zeros, at most 2 facilities are opened

---

## Prompts sequence

The prompts to use in Claude Code are listed in PROMPTS.md.