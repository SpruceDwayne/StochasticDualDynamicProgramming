# Prompts for implementing the facility location DDU problem

Use these prompts **in order** in Claude Code (VS Code extension). Each prompt builds on the previous one. Copy `SPEC.md` and `instance_data.json` into your project directory before starting.

---

## Prompt 1 — Data loading

```
Read SPEC.md for full context. Implement `src/data.jl`: a function
`load_instance(path::String)` that reads `instance_data.json` and returns a
named tuple or struct with all fields converted to 1-based Julia indexing.
Specifically:
- facility_zones: Vector{Int} (add 1 to each)
- zone_order: Dict{Int, Vector{Int}} (keys and values both +1)
- activation_regions: Vector{Set{Int}} (each inner list → Set, values +1)
- profit_matrix, dist_matrix: Matrix{Float64}
- alpha, beta vectors: Vector{Float64}

Use JSON.jl to parse. Write a short test in `test/test_data.jl` that verifies:
1. Zone 1 contains facilities [1,2,6,7,8]
2. profit_matrix size is (8,15) with min ≈ 183 and max ≈ 394
3. Customer 1 base mean = 10 * α[1]/(α[1]+β[1]) ≈ 1.5
```

---

## Prompt 2 — DDU distributions

```
Read SPEC.md section "DDU mapping" for the exact formulas. Implement
`src/distributions.jl` with:

1. `compute_demand_distributions(inst, zone_active::Vector{Bool})` that returns
   a Dict{Int, Vector{Float64}} mapping customer index → PMF vector (length 11)
   over {0,...,10}. Use Distributions.jl BetaBinomial.

2. `region_to_zone_active(inst, d::Int)` that returns the zone activation
   vector for region d.

3. `identify_region(inst, x::Vector{Int}, b::Vector{Int})` that returns the
   region index d given facility decisions x and habit state b.

4. `sample_scenarios(inst, d::Int; N=100)` that returns a matrix of N demand
   scenarios (each row is a vector in {0,...,10}^15) sampled from the joint
   distribution for region d, plus a probability vector (uniform 1/N).

Test in `test/test_distributions.jl`:
- x=zeros(8), b=zeros(3) → all zones inactive → region 1 → customer 1 mean ≈ 1.5
- x=ones(8), b=ones(3) → all zones active → region 8 → customer 8 mean ≈ 9.38
- identify_region with only facility 1 open → region 2 (zone 1 only)
```

---

## Prompt 3 — Stage subproblem

```
Read SPEC.md section "Stage-t subproblem" for the full formulation. Implement
`src/subproblem.jl` with a function `build_stage_problem(inst, t, x_prev, b_prev, xi; solver=HiGHS.Optimizer)` that returns a JuMP model for the stage-t problem.

Decision variables:
- x[1:8] Binary (facility config)
- w[1:8, 1:15] ≥ 0 (demand served)
- b[1:3] Binary (habit state)
- a[1:3] Binary (zone activation)
- δ[1:8] Binary (region indicator)
- θ (cost-to-go, free variable; fix to 0 at terminal stage)

Objective: min  -Σ p[i,j]*w[i,j] + Σ O*(x[i]-x_prev[i]) + θ

Constraints as in SPEC.md. For the region-activation linking, use:
  a[z] = Σ_{d : z ∈ S_d} δ[d]   ∀ z

Do NOT add any cuts yet — just the base subproblem. The solver will add cuts.

Test in `test/test_subproblem.jl`:
- Build stage 1 with x_prev=zeros, b_prev=zeros, xi=5*ones(15)
- Solve it with θ fixed to 0
- Verify the model is feasible and objective is negative (profit)
- Verify x monotonicity: with x_prev=[1,0,...,0], solution has x[1]=1
```

---

## Prompt 4 — Solver integration

```
Read SPEC.md section "What the solver expects" and look at the existing DDU-SDDP
solver code to understand its interface. Now implement `src/run.jl` that:

1. Loads the instance via load_instance()
2. Wraps the subproblem builder, distribution sampler, and region identifier
   into whatever interface the solver expects
3. Configures the solver for T=4 stages, with appropriate big-M values
4. Runs DDU-SDDP and prints the lower bound per iteration

For big-M: use M_d = max possible |v + βᵀx| over x ∈ {0,1}^8 plus a margin.
A safe value is M_d = 100000 (since max single-stage profit ≈ 394*120 ≈ 47000).

Start with N_samples=50 scenarios per region and max_iterations=100.

Print:
- Lower bound per iteration
- Final first-stage decision (which facilities to open)
- Active region at stage 1
- Total wall time
```

---

## Prompt 5 — Verification and comparison

```
Now let's verify the results and produce comparison data for the lecture notes.

1. Run the same instance but with STANDARD SDDiP (ignoring DDU): treat demand
   as exogenous using the distribution from region 1 (no facilities open) for
   all stages. This gives the "ignoring DDU" baseline.

2. Run DDU-SDDP as before (this is the "DDU-aware" policy).

3. Evaluate both policies out-of-sample: simulate 1000 sample paths using the
   TRUE DDU distributions (i.e., demand depends on decisions). For each path,
   record total profit.

4. Create a summary table:
   | Policy     | Avg profit | Std dev | First-stage facilities opened |
   
5. Create a convergence plot: iteration vs lower bound for both methods.

Save the table as a CSV and the plot as a PNG.
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
```