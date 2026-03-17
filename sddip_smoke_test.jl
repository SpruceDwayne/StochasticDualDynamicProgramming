using Pkg
Pkg.activate(joinpath(@__DIR__, "SDDPBAPE"))
using SDDPBAPE
using JuMP, HiGHS
using LinearAlgebra

# ── Parameters ────────────────────────────────────────────────────────────────
const Gmax = 10.0
const d1   = 6.0
const SU   = 2.0
const F    = 0.5
const c    = 1.0
const M    = 100.0

const scenarios = [4.0, 9.0]   # d2 for L and H
const probs     = [0.5, 0.5]

# ── Stage 1 builder ───────────────────────────────────────────────────────────
# State: x1 ∈ {0,1} (generator ON/OFF).  No parent state to copy at t=1.
# fix_state is ignored here (no copy constraint at stage 1).
function build_stage1(t, vf_next, ω; fix_state)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x1, Bin)          # state: generator ON
    @variable(model, u1, Bin)          # startup
    @variable(model, g1  >= 0)         # generation
    @variable(model, shed1 >= 0)       # load shed
    @variable(model, θ >= 0)           # cost-to-go approximation

    # power balance
    @constraint(model, g1 + shed1 == d1)
    # capacity
    @constraint(model, g1 <= Gmax * x1)
    # startup logic (start from OFF)
    @constraint(model, u1 >= x1)

    # epigraph: θ ≥ cuts from vf_next evaluated at x1
    for cut in vf_next.cuts
        @constraint(model, θ >= cut.α + cut.β[1] * x1)
    end

    @objective(model, Min, SU*u1 + F*x1 + c*g1 + M*shed1 + θ)

    misc = Dict{Symbol,Any}(
        :x_next => [x1],   # next state is x1 itself (the binary)
        :x_state => [x1],  # for compute_cut! (not used in SDDiP backward but required)
    )
    return model, [x1], θ, misc
end

# ── Stage 2 builder ───────────────────────────────────────────────────────────
# State: x2 ∈ {0,1}.  z2 ∈ [0,1] copies parent x1.
# fix_state = [x1_val] is used to set z2 == x1_val initially.
function build_stage2(t, vf_next, ω; fix_state)
    d2 = ω   # demand for this scenario

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x2, Bin)          # state decision
    @variable(model, u2, Bin)          # startup
    @variable(model, g2  >= 0)
    @variable(model, shed2 >= 0)

    # SDDiP copy variable: z2 ∈ [0,1] copies parent binary state
    @variable(model, 0 <= z2 <= 1)

    # Copy constraint: z2 == fix_state[1]  (will be removed and replaced by JuMP.fix)
    @constraint(model, z2 == fix_state[1])

    # Power balance
    @constraint(model, g2 + shed2 == d2)
    # Capacity
    @constraint(model, g2 <= Gmax * x2)
    # Startup tied to copied parent state
    @constraint(model, u2 >= x2 - z2)

    # Terminal stage: θ = 0 (no cost-to-go)
    @variable(model, θ == 0)

    @objective(model, Min, SU*u2 + F*x2 + c*g2 + M*shed2 + θ)

    misc = Dict{Symbol,Any}(
        :z_vars  => [z2],  # SDDiP: the copy variable
        :x_next  => [x2],
        :x_state => [x2],
    )
    return model, [x2], θ, misc
end

# ── Stage setup ───────────────────────────────────────────────────────────────
# Stage 1: IID, no randomness (ω is a dummy)
stage1 = Stage(
    1, 1,
    build_stage1,
    () -> nothing,       # sampler
    _ -> 1.0,           # weight
    (t, ctx) -> ([nothing], [1.0]),   # children: just one dummy child
    (t, ctx, ω) -> nothing,           # next_ctx
    x -> x,             # node_key
)

# Stage 2: two scenarios
stage2 = Stage(
    2, 1,
    build_stage2,
    () -> scenarios[rand(1:2)],               # sampler
    _ -> 1.0,
    (t, ctx) -> (scenarios, probs),           # children
    (t, ctx, ω) -> nothing,
    x -> x,
)

# ── Assemble model ────────────────────────────────────────────────────────────
m = SDDP([stage1, stage2])
x0 = [0.0]   # start from generator OFF

config = SDDiPConfig(cut_type = :SB)

println("="^60)
println("SDDiP Smoke Test: 2-Stage Binary Unit Commitment")
println("="^60)

# ── Run one iteration manually so we can inspect each step ────────────────────
println("\n--- Forward pass ---")
fwd = forward_pass_online!(m; S=1, x0=x0, ctx0=nothing)
x1_val = fwd.x_state[1][1][1]
println("  x1 (generator ON at stage 1): $x1_val  (should be binary)")
@assert x1_val ≈ 0.0 || x1_val ≈ 1.0  "x1 is not binary! got $x1_val"
println("  PASS: x1 is binary")

println("\n--- Backward pass (one manual child solve to inspect π) ---")
vf_next = m.V[2]   # empty at start

# pick scenario L (d2=4.0) and manually drive get_or_build_model! + _get_lp_dual
x_support = fwd.x_state[1][1]   # x1 from forward pass
println("  x_support (parent state for stage-2 backward): $x_support")

model2, _, _, misc2 = SDDPBAPE.get_or_build_model!(m, 2, vf_next, 4.0, nothing, x_support)
@assert haskey(misc2, :z_vars) "misc[:z_vars] missing in stage-2 builder"
println("  PASS: misc[:z_vars] present")

z_vars = misc2[:z_vars]
println("  z_vars fixed to: $(JuMP.fix_value(z_vars[1]))")
@assert JuMP.fix_value(z_vars[1]) ≈ x_support[1]  "z not fixed to x_parent!"
println("  PASS: z2 fixed to x1_val")

println("\n--- LP relaxation dual extraction ---")
π_lp, lp_obj = SDDPBAPE._get_lp_dual(model2, z_vars)
println("  LP obj = $lp_obj,  π_LP = $π_lp")
println("  PASS: LP dual extracted without error")

println("\n--- Lagrangian subproblem (SB step) ---")
L_val, z_sol = SDDPBAPE._solve_lagrangian_subproblem!(model2, z_vars, π_lp, x_support)
println("  L(π_LP) = $L_val,  z_sol = $z_sol")
println("  PASS: Lagrangian subproblem solved")

# verify z is re-fixed after the call
@assert JuMP.is_fixed(z_vars[1]) "z_vars not re-fixed after Lagrangian subproblem!"
println("  PASS: z_vars re-fixed after subproblem")

println("\n--- compute_sddip_cut! (full SB cut) ---")
pairs = compute_sddip_cut!(model2, z_vars, x_support, config)
@assert length(pairs) == 1  "Expected 1 cut pair for :SB, got $(length(pairs))"
α, β = pairs[1]
println("  Cut: θ ≥ $(round(α,digits=4)) + $(round(β[1],digits=4))*x1")
println("  PASS: SB cut generated")

println("\n--- Full backward pass (adds SDDiP cut to V[2]) ---")
cuts_before = length(m.V[2].cuts)
backward_pass_sddip!(m; fwd=fwd, config=config, iter=1, force_every=1)
cuts_after = length(m.V[2].cuts)
println("  Cuts in V[2] before: $cuts_before,  after: $cuts_after")
@assert cuts_after > cuts_before  "No SDDiP cut was added to V[2]!"
println("  PASS: SDDiP cut added to V[2]")

println("\n--- run_sddip! for 5 iterations ---")
result = run_sddip!(m; x0=x0, config=config, max_iter=5, patience=5,
                    force_every=1, evaluate_index=2)
println("  Completed $(result.iters) iterations, total cuts: $(sum(result.cuts_per_stage))")
println("  Cuts per stage: $(result.cuts_per_stage)")
println("  PASS: run_sddip! completed without error")

println("\n" * "="^60)
println("ALL SMOKE TEST CHECKS PASSED")
println("="^60)
