# Facility Location with Decision-Dependent Uncertainty — Implementation Spec

## What this document is

This is a specification for implementing a **multistage facility location problem with decision-dependent uncertainty (DDU)** in Julia. The problem will be solved using an existing DDU-SDDP/SDDiP solver. The task is to build the **problem instance**, **stage subproblems**, and **DDU mapping** so they can be passed to the solver.

---

## Problem overview

A firm decides which facilities to open over T stages. Opening facilities affects future customer demand (decision-dependent uncertainty). The firm earns revenue by serving demand, pays fixed costs to open new facilities, and wants to maximize total expected profit.

The key DDU mechanism: facilities are grouped into **zones**. Which zones are "active" (have ≥1 open facility) determines the probability distribution of next-stage demand. Additionally, a **habit state** b_z ∈ {0,1} per zone tracks whether a customer base has ever been established there; once set, it persists.

---

## Sets and indices

| Symbol | Description | Size |
|--------|-------------|------|
| I | Facilities | 8 (indexed 1:8) |
| J | Customers | 15 (indexed 1:15) |
| Z | Zones | 3 (indexed 1:3) |
| D | Regions (activation patterns) | 8 (powerset of Z, indexed 1:8) |
| T | Stages | User-specified (e.g. 4) |

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
| n | 10 | Maximum demand per customer (BetaBinomial support: {0,...,10}) |
| s | 6 | BetaBinomial scale parameter |

### Profit matrix

```
p[i,j] = R - c * dist(i,j)    for all (i,j) ∈ I × J
```

where `dist(i,j)` is the Euclidean distance between facility i and customer j. All entries are positive (range ≈ 183 to 394). The full profit matrix and coordinates should be loaded from `instance_data.json`.

---

## State variables

The full state entering stage t is `(x_{t-1}, b_{t-1})`:

- **x_{t-1} ∈ {0,1}^8**: which facilities are currently open
- **b_{t-1} ∈ {0,1}^3**: which zones have established customer habits

Total state dimension: 11 binary variables.

Initial state: `x_0 = 0, b_0 = 0` (nothing open, no habits).

---

## Stage-t subproblem

### Decision variables

| Variable | Type | Dimension | Description |
|----------|------|-----------|-------------|
| x_t | Binary | 8 | Facility configuration after stage t |
| w_{ij} | Continuous ≥ 0 | 8 × 15 | Demand from customer j served by facility i |
| b_t | Binary | 3 | Habit state after stage t |
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

**Zone activation:**
```
a[z] ≤ Σ_{i ∈ I_z} x[i,t]         ∀ z ∈ Z     (if no facility open, zone inactive)
a[z] ≥ x[i,t]                      ∀ i ∈ I_z, z ∈ Z  (if any facility open, zone active)
a[z] ∈ {0,1}
```

Equivalently: `a[z] = 1 iff Σ_{i ∈ I_z} x[i,t] ≥ 1`.

**Habit state evolution (linearized max):**
```
b[z,t] ≥ b[z,t-1]                  ∀ z ∈ Z     (habits persist)
b[z,t] ≥ a[z]                      ∀ z ∈ Z     (new activation creates habit)
b[z,t] ≤ b[z,t-1] + a[z]           ∀ z ∈ Z     (tight: equals max)
b[z,t] ∈ {0,1}
```

**Region activation:**

Each region d corresponds to a specific subset S_d ⊆ Z of active zones. The constraint `δ[d] = 1 iff (a[z] = 1 ∀ z ∈ S_d) and (a[z] = 0 ∀ z ∉ S_d)` can be linearized as:

```
Σ_d δ[d] = 1                                    (exactly one region active)
δ[d] ∈ {0,1}                                    ∀ d ∈ D

# Linking δ to a:
# For each region d with active zone set S_d:
#   if δ[d] = 1, then a[z] = 1 for z ∈ S_d and a[z] = 0 for z ∉ S_d
#
# This is enforced by:
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

Given a state-decision pair `(x_t, b_t)` at stage t, the next-stage demand is sampled as follows.

### Step 1: Determine effective zone activation

```
zone_active[z] = max(a[z], b[z,t])    ∀ z
```

where `a[z] = 1{Σ_{i ∈ I_z} x[i,t] ≥ 1}`.

### Step 2: Compute demand distribution per customer

For each customer j:

1. Look up `zone_order[j]` — the 3 zones sorted by distance from customer j to zone centroid (loaded from `instance_data.json`, field `zone_order_per_customer`). **Note:** the JSON keys are 0-indexed strings; convert to 1-indexed.

2. Compute effective mean:
   ```
   base_mean[j] = 10 * α[j] / (α[j] + β[j])
   mean[j] = base_mean[j]
   for k = 1, 2, 3:
       z = zone_order[j][k]
       if zone_active[z] == 1:
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

### Step 3: Discretize for the solver

The solver needs a finite set of scenarios with probabilities for each region d. Precompute the full PMF vectors (length 11 each) for all 15 customers, for each of the 8 regions. Customer demands are independent conditional on the region, so a scenario is a vector in `{0,...,10}^15`.

**Important:** for tractability, you will likely want to sample a moderate number of scenarios per region rather than enumerate all `11^15` possibilities. A reasonable approach: for each region, draw N_samples scenarios from the joint distribution (product of independent BetaBinomials) and assign equal probability 1/N_samples to each.

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

1. **Instance data struct** containing all parameters, sets, and precomputed data.

2. **A function that builds the stage-t subproblem** as a JuMP model, given:
   - Stage index t
   - Incoming state (x_{t-1}, b_{t-1})
   - Uncertainty realization ξ_t
   - Current cut collection for each region

3. **A function that returns demand distributions** (scenario set + probabilities) for a given region d. This is called during the forward pass to sample next-stage uncertainty.

4. **Region identification function**: given a solution (x_t, b_t), return which region d is active.

Consult the solver's existing interface/API and match these to whatever types and function signatures it expects.

---

## Suggested file structure

```
facility_location/
├── instance_data.json          # Generated instance (already exists)
├── problem_src/
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
- [ ] For x = zeros(8), b = zeros(3): customer 1 mean demand = 1.5, customer 15 mean = 8.5
- [ ] For x = ones(8), b = ones(3): customer 8 mean demand ≈ 9.38
- [ ] Region activation: x = [1,0,0,0,0,0,0,0] (only fac 1 open → zone 1 active) → region 2
- [ ] Single-stage subproblem solves and returns sensible profit
- [ ] Opening cost is only charged for *newly* opened facilities

---

## Prompts sequence

The prompts to use in Claude Code are listed in PROMPTS.md.