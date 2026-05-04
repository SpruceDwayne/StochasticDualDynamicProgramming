using JuMP
using HiGHS

include(joinpath(@__DIR__, "..", "problem_src", "subproblem.jl"))

const INSTANCE_PATH = joinpath(@__DIR__, "..", "..", "instance_data.json")
inst = load_instance(INSTANCE_PATH)

# ── Test 1: feasibility and negative objective (profit) ───────────────────────
x0 = zeros(Int, inst.num_facilities)
b0 = zeros(Int, inst.num_zones)
xi = fill(5.0, inst.num_customers)

m1 = build_stage_problem(inst, 1, x0, b0, xi)
fix(m1[:θ], 0.0; force=true)
optimize!(m1)

@assert termination_status(m1) == MOI.OPTIMAL "Test 1: expected OPTIMAL, got $(termination_status(m1))"
obj1 = objective_value(m1)
@assert obj1 < 0 "Test 1: expected negative objective (profit), got $obj1"
println("Test 1 passed: feasible, objective = $(round(obj1; digits=2))")

# ── Test 2: facility monotonicity ────────────────────────────────────────────
x_prev2 = zeros(Int, inst.num_facilities)
x_prev2[1] = 1   # facility 1 already open

m2 = build_stage_problem(inst, 1, x_prev2, b0, xi)
fix(m2[:θ], 0.0; force=true)
optimize!(m2)

@assert termination_status(m2) == MOI.OPTIMAL "Test 2: expected OPTIMAL"
x2_sol = round.(Int, value.(m2[:x]))
@assert x2_sol[1] == 1 "Test 2: monotonicity violated — x[1] should be 1, got $(x2_sol[1])"
println("Test 2 passed: x[1]=1 enforced by monotonicity, x = $x2_sol")

# ── Test 3: per-stage opening budget ≤ 2 ─────────────────────────────────────
m3 = build_stage_problem(inst, 1, x0, b0, xi)
fix(m3[:θ], 0.0; force=true)
optimize!(m3)

@assert termination_status(m3) == MOI.OPTIMAL "Test 3: expected OPTIMAL"
x3_sol = round.(Int, value.(m3[:x]))
opened = sum(x3_sol)
@assert opened <= 2 "Test 3: budget violated — opened $opened facilities (max 2)"
println("Test 3 passed: opened $opened facilities (budget ≤ 2), x = $x3_sol")

# ── Test 4: is_terminal fixes θ = 0 ──────────────────────────────────────────
m4 = build_stage_problem(inst, 4, x0, b0, xi; is_terminal=true)
optimize!(m4)

@assert termination_status(m4) == MOI.OPTIMAL "Test 4: expected OPTIMAL"
θ4 = value(m4[:θ])
@assert isapprox(θ4, 0.0; atol=1e-6) "Test 4: expected θ=0 at terminal stage, got $θ4"
println("Test 4 passed: terminal stage θ = $θ4")

println("All subproblem tests passed.")
