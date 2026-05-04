# Prompts for implementing the facility location DDU problem

Use these prompts **in order** in Claude Code (VS Code extension). Each prompt builds on the previous one. Copy `SPEC.md` and `instance_data.json` into your project directory before starting.

---

## Prompt 1 — Data loading

```
Read SPEC.md for full context. Implement `problem_src/data.jl`: a function
`load_instance(path::String)` that reads `instance_data.json` and returns a
named tuple or struct with all fields converted to 1-based Julia indexing.
Specifically:
- facility_zones: Vector{Int} (add 1 to each)
- zone_order: Dict{Int, Vector{Int}} (keys and values both +1)
- activation_regions: Vector{Set{Int}} (each inner list → Set, values +1)
- profit_matrix, dist_matrix: Matrix{Float64}
- alpha, beta vectors: Vector{Float64}

Also store derived data:
- zone_facilities: Dict{Int, Vector{Int}} mapping zone → list of facility indices

Use JSON.jl to parse. Write a short test in `test/test_data.jl` that verifies:
1. Zone 1 contains facilities [1,2,6,7,8]
2. profit_matrix size is (8,15) with min ≈ 183 and max ≈ 394
3. Customer 1 base mean = 10 * α[1]/(α[1]+β[1]) ≈ 1.5
```

---

## Prompt 2 — DDU distributions

```
Read SPEC.md sections "DDU mapping" and "SAA discretization" for the exact
formulas and approach. Implement `problem_src/distributions.jl` with:

1. `compute_demand_pmfs(inst, d::Int)` that returns a Matrix{Float64} of size
   (15, 11) where row j is the BetaBinomial PMF for customer j under region d.
   Use Distributions.jl BetaBinomial.

2. `region_to_zone_active(inst, d::Int)` that returns the zone activation
   vector (Vector{Bool}) for region d.

3. `identify_region(inst, x::Vector{Int})` that returns the region index d
   given facility configuration x. Compute zone activation from x, then find
   the matching region.

4. `generate_saa_scenarios(inst; N=50, seed=42)` that generates the FIXED
   scenario sets used throughout the entire algorithm. Returns a Dict{Int,
   Matrix{Int}} mapping region d → matrix of size (N, 15), where each row is
   a demand realization sampled independently per customer from their PMF for
   region d. All regions use the same N. Fix the random seed for
   reproducibility. This function is called ONCE before the algorithm starts.

Test in `test/test_distributions.jl`:
- x=zeros(8) → no zones active → region 1 → customer 1 mean ≈ 1.5
- x=ones(8) → all zones active → region 8 → customer 8 mean ≈ 9.38
- identify_region with only facility 1 open → region 2 (zone 1 only)
- generate_saa_scenarios returns Dict with 8 keys, each value is (50, 15) matrix
- sample mean of scenarios for region 8, customer 8 ≈ 9.38 (within sampling noise)
```

---

## Prompt 3 — Stage subproblem

```
Read SPEC.md section "Stage-t subproblem" for the full formulation. Implement
`problem_src/subproblem.jl` with a function `build_stage_problem(inst, t, x_prev, xi;
solver=HiGHS.Optimizer)` that returns a JuMP model for the stage-t problem.

Decision variables:
- x[1:8] Binary (facility config)
- w[1:8, 1:15] ≥ 0 (demand served)
- a[1:3] Binary (zone activation)
- δ[1:8] Binary (region indicator)
- θ (cost-to-go, free variable; fix to 0 at terminal stage)

Objective: min  -Σ p[i,j]*w[i,j] + Σ O*(x[i]-x_prev[i]) + θ

Constraints as in SPEC.md. For the region-activation linking, use:
  a[z] = Σ_{d : z ∈ S_d} δ[d]   ∀ z

Include the per-stage opening budget constraint:
  Σ_i (x[i] - x_prev[i]) ≤ k     (k=2, from inst)

Do NOT add any cuts yet — just the base subproblem. The solver will add cuts.

Test in `test/test_subproblem.jl`:
- Build stage 1 with x_prev=zeros, xi=5*ones(15)
- Solve it with θ fixed to 0
- Verify the model is feasible and objective is negative (profit)
- Verify x monotonicity: with x_prev=[1,0,...,0], solution has x[1]=1
- Verify budget: with x_prev=zeros, verify sum(x) ≤ 2 in the solution
```

---

## Prompt 4 — Solver integration

```
Read SPEC.md sections "What the solver expects" and "SAA discretization", and
look at the existing DDU-SDDP solver code to understand its interface. Now
implement `problem_src/run.jl` that:

1. Loads the instance via load_instance()
2. Generates the fixed SAA scenario sets via generate_saa_scenarios(inst; N=20)
   — this is done ONCE before the algorithm starts
3. Wraps the subproblem builder, scenario sets, and region identifier into
   whatever interface the solver expects
4. Configures the solver for T=4 stages, with appropriate big-M values
5. Runs DDU-SDDP and prints the lower bound per iteration

The SAA scenarios are fixed throughout all iterations:
- Forward pass: for each stage, once the active region d is identified, sample
  uniformly from the N precomputed scenarios for region d
- Backward pass: compute the expected cost-to-go by solving the subproblem for
  ALL N scenarios of the active region and averaging — this is exact over the
  SAA support

For big-M: use M_d = max possible |v + βᵀx| over x ∈ {0,1}^8 plus a margin.
A safe value is M_d = 100000 (since max single-stage profit ≈ 394*120 ≈ 47000).

Run with max_iterations=200.

Print:
- Lower bound per iteration
- Final first-stage decision (which 2 facilities are opened)
- Active region at stage 1
- Total wall time
```

---

## Prompt 5 — Verification and comparison

```
Now let's verify the results and produce comparison data for the lecture notes. For this make a new file run_comparison.jl. You may need to recap the problem by reading SPEC.md.

1. Run the same instance but with STANDARD SDDiP (ignoring DDU): treat demand
   as exogenous using the distribution from region 1 (no facilities open) for
   all stages. This gives the "ignoring DDU" baseline.

2. Run DDU-SDDP as before (this is the "DDU-aware" policy).

3. Evaluate both policies out-of-sample: simulate 1000 sample paths using the
   TRUE DDU distributions (i.e., demand depends on decisions). For each path,
   record total profit.

4. Create a summary table:
   | Policy     | Avg profit | Std dev | First-stage facilities opened |

5. Create a convergence plot: 'iteration vs lower bound' AND 'time vs lower bound' for both methods. 

Save the table as a CSV, the data used in the plot as a CSV, and the plot as a PNG. 

For this it may (or may not) be useful to see how convergence is examined in examples/complexity_tests.jl
```

---

## Prompt 6 — Results for lecture notes

```
Generate publication-quality output for the lecture notes:

1. A convergence plot (PDF or PNG, 300 dpi) showing DDU-SDDP vs standard SDDiP
   lower bounds over iterations.

2. A table (LaTeX format) with columns:
   - Method, Iterations to converge, Lower bound, Simulated avg profit (1000 paths),
   Simulated std dev, Facilities opened at t=1

3. A bar chart or table showing which facilities each method opens at each
   stage (averaged or modal over the simulated paths).

Print the LaTeX table to stdout so I can paste it into the chapter.

save all data used in the plots as a csv file such that I can reconstruct them myself manually
```