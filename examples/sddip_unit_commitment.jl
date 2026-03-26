using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
using SDDPBAPE
using JuMP, HiGHS, LinearAlgebra, Printf, Random, Statistics

# ============================================================================
# Problem: 2-Stage Stochastic Binary Unit Commitment
# ============================================================================
#
# A power plant owner must decide each period whether to keep a generator
# ON (x=1) or OFF (x=0).  Period 1 is deterministic; period 2 has random
# demand that is only revealed after the period-1 decision is locked in.
#
# Decision variables per stage:
#   x_t  ∈ {0,1}   generator ON/OFF state (also the binary state variable)
#   u_t  ∈ {0,1}   1 if the generator starts up this period
#   g_t  ≥ 0       energy generated [MWh]
#   shed_t ≥ 0     unmet demand (load shedding) [MWh]
#
# Per-stage cost:
#   SU * u_t  — startup cost ($/startup)
#   F  * x_t  — fixed operating cost ($/period when on)
#   c  * g_t  — variable generation cost ($/MWh)
#   M  * shed_t — load-shedding penalty (very expensive)
#
# Constraints:
#   g_t + shed_t = d_t          power balance
#   g_t ≤ Gmax * x_t            capacity (can only generate if ON)
#   u_t ≥ x_t - x_{t-1}        startup logic (1 if turning on)
#
# Copy constraint (required by SDDiP):
#   z_t ∈ [0,1]  is a continuous variable that copies parent binary state x_{t-1}.
#   The constraint z_t = x_{t-1} is dualised to generate Lagrangian cuts.
#
# Stochasticity:
#   d_1 = 6.0 MW (deterministic)
#   d_2 ∈ {4.0, 9.0} MW with equal probability (revealed in period 2)
#
# SDDiP with Lagrangian cuts:
#   At each backward pass, the copy constraint z_2 = x_1 is relaxed via a
#   multiplier π.  Subgradient ascent on g(π) = L(π) + π·x_1 (where L(π) is
#   the Lagrangian subproblem minimum) produces a valid, tight cut
#     θ ≥ L(π*) + π*·x_1
#   that is added to the stage-1 value-function approximation V[2](x_1).
# ============================================================================

# ── Parameters ────────────────────────────────────────────────────────────────
const Gmax = 10.0   # generator capacity [MWh]
const d1   = 6.0    # stage-1 demand [MW]
const SU   = 2.0    # startup cost [$/startup]
const F    = 0.5    # fixed ON cost [$/period]
const c    = 1.0    # variable cost [$/MWh]
const M    = 100.0  # load-shedding penalty [$/MWh]

const scenarios = [4.0, 9.0]  # stage-2 demand scenarios [MW]
const probs     = [0.5, 0.5]  # equal probabilities

# ── Stage builders ────────────────────────────────────────────────────────────
#
# Stage 1: no parent binary state to copy.
#   fix_state = x0 (initial state, ignored here — x1 is the decision, not fixed).
#   misc[:x_next] points to x1 so the forward pass advances the state correctly.
#
function build_stage1(t, vf_next, ω; fix_state)
    model = Model(HiGHS.Optimizer); set_silent(model)

    @variable(model, x1, Bin)        # binary state decision
    @variable(model, u1, Bin)        # startup indicator
    @variable(model, g1    >= 0)     # generation
    @variable(model, shed1 >= 0)     # unmet demand
    @variable(model, θ     >= 0)     # cost-to-go approximation

    @constraint(model, g1 + shed1 == d1)
    @constraint(model, g1 <= Gmax * x1)
    @constraint(model, u1 >= x1)              # start from OFF (x_{t-1} = 0)

    # Epigraph cuts from V[2] (added by the algorithm; re-applied here each build)
    for cut in vf_next.cuts
        @constraint(model, θ >= cut.α + cut.β[1] * x1)
    end

    @objective(model, Min, SU*u1 + F*x1 + c*g1 + M*shed1 + θ)

    misc = Dict{Symbol,Any}(
        :x_next  => [x1],   # state passed to stage 2
        :x_state => [x1],
    )
    return model, [x1], θ, misc
end

# Stage 2: terminal stage; x2 ∈ {0,1}, z2 ∈ [0,1] copies x1.
#   fix_state = [x1_val] is used to initialise the copy constraint z2 = x1_val.
#   misc[:z_vars] = [z2] tells SDDiP which variable is the copy variable.
#
function build_stage2(t, vf_next, ω; fix_state)
    d2 = ω
    model = Model(HiGHS.Optimizer); set_silent(model)

    @variable(model, x2, Bin)
    @variable(model, u2, Bin)
    @variable(model, g2    >= 0)
    @variable(model, shed2 >= 0)

    # SDDiP copy variable: z2 ∈ [0,1] copies the parent binary state x1.
    # The copy constraint z2 = fix_state[1] is replaced by JuMP.fix internally.
    @variable(model, 0 <= z2 <= 1)
    @constraint(model, z2 == fix_state[1])

    @constraint(model, g2 + shed2 == d2)
    @constraint(model, g2 <= Gmax * x2)
    @constraint(model, u2 >= x2 - z2)   # startup only if turning on from OFF

    @variable(model, θ == 0)            # terminal: no further cost-to-go

    @objective(model, Min, SU*u2 + F*x2 + c*g2 + M*shed2 + θ)

    misc = Dict{Symbol,Any}(
        :z_vars  => [z2],   # SDDiP: the continuous copy variable
        :x_next  => [x2],
        :x_state => [x2],
    )
    return model, [x2], θ, misc
end

# ── Assemble SDDP model ───────────────────────────────────────────────────────
stage1 = Stage(
    1, 1, build_stage1,
    () -> nothing,                            # no randomness at stage 1
    _ -> 1.0,
    (t, ctx) -> ([nothing], [1.0]),           # single dummy child
    (t, ctx, ω) -> nothing,
    x -> x,
)

stage2 = Stage(
    2, 1, build_stage2,
    () -> scenarios[rand(1:2)],               # sampler: draw a scenario
    _ -> 1.0,
    (t, ctx) -> (scenarios, probs),           # children: both scenarios
    (t, ctx, ω) -> nothing,
    x -> x,
)

m  = SDDP([stage1, stage2])
x0 = [0.0]   # initial state: generator OFF

# ── Train SDDiP with Lagrangian cuts ─────────────────────────────────────────
config = SDDiPConfig(
    cut_type       = :lagrangian,
    lag_tol        = 1e-4,
    lag_max_iter   = 200,
    step_size_init = 1.0,
    step_decay     = 0.95,
)

println("="^60)
println("2-Stage Binary Unit Commitment  —  SDDiP (Lagrangian cuts)")
println("="^60)
println()
println("Training SDDiP (patience = 10 iterations without improvement)…")
println()

result = run_sddip!(
    m;
    x0            = x0,
    config        = config,
    max_iter      = 200,
    patience      = 10,
    force_every   = 5,
    evaluate_index = 2,          # monitor V[2] for convergence
    cut_atol      = 1e-8,
)

println()
@printf "Converged in %d iterations.  Cuts in V[2]: %d\n" result.iters result.cuts_per_stage[2]

# ── V[2] cut coefficients ─────────────────────────────────────────────────────
println()
println("─"^60)
println("Stage-2 value function V[2](x1) = max_k { α_k + β_k · x1 }")
println("  (each row is one Lagrangian cut)")
println("─"^60)
@printf "  %6s  %8s  %8s\n" "cut" "α" "β"
for (k, cut) in enumerate(m.V[2].cuts)
    @printf "  %6d  %8.4f  %8.4f\n" k cut.α cut.β[1]
end

# Evaluate the approximation at x1 ∈ {0, 1}
v2_at_0, _ = evaluate(m.V[2], [0.0])
v2_at_1, _ = evaluate(m.V[2], [1.0])
println()
@printf "  V[2](x1=0) ≈ %.4f\n" v2_at_0
@printf "  V[2](x1=1) ≈ %.4f\n" v2_at_1

# ── Optimal first-stage decision ──────────────────────────────────────────────
# Build a fresh stage-1 model with all learned V[2] cuts, leaving x1 FREE
# so the solver can choose the optimal binary decision.
println()
println("─"^60)
println("Optimal stage-1 decisions  (starting from generator OFF)")
println("─"^60)

let model1 = Model(HiGHS.Optimizer)
    set_silent(model1)

    @variable(model1, x1,    Bin)
    @variable(model1, u1,    Bin)
    @variable(model1, g1    >= 0)
    @variable(model1, shed1 >= 0)
    @variable(model1, θ     >= 0)

    @constraint(model1, g1 + shed1 == d1)
    @constraint(model1, g1 <= Gmax * x1)
    @constraint(model1, u1 >= x1)

    for cut in m.V[2].cuts
        @constraint(model1, θ >= cut.α + cut.β[1] * x1)
    end

    @objective(model1, Min, SU*u1 + F*x1 + c*g1 + M*shed1 + θ)
    optimize!(model1)

    x1_opt    = round(Int, value(x1))
    u1_opt    = round(Int, value(u1))
    g1_opt    = value(g1)
    shed1_opt = value(shed1)
    θ_opt     = value(θ)
    stage1_op = SU*u1_opt + F*x1_opt + c*g1_opt + M*shed1_opt

    @printf "  Generator ON (x1)  : %d\n"     x1_opt
    @printf "  Startup    (u1)    : %d\n"     u1_opt
    @printf "  Generation (g1)    : %.2f MW\n" g1_opt
    @printf "  Load shed  (shed1) : %.2f MW\n" shed1_opt
    @printf "  Stage-1 op. cost   : %.4f\n"   stage1_op
    @printf "  Continuation (θ)   : %.4f  [≈ E[V[2](x1)]]\n" θ_opt
    @printf "  Stage-1 obj. total : %.4f\n"   objective_value(model1)
end

# ── Expected cost under the optimal policy ─────────────────────────────────
# 1. Exact expectation: solve stage-2 for each scenario with x1 fixed to
#    the optimal decision, then take the probability-weighted average.
# 2. Monte Carlo estimate as a sanity check.
println()
println("─"^60)
println("Expected total cost under the optimal policy")
println("─"^60)

# Recover x1_opt from a fresh solve (same model as above, kept self-contained)
x1_opt_val = let model1 = Model(HiGHS.Optimizer)
    set_silent(model1)
    @variable(model1, x1, Bin); @variable(model1, u1, Bin)
    @variable(model1, g1 >= 0); @variable(model1, shed1 >= 0)
    @variable(model1, θ >= 0)
    @constraint(model1, g1 + shed1 == d1)
    @constraint(model1, g1 <= Gmax * x1)
    @constraint(model1, u1 >= x1)
    for cut in m.V[2].cuts
        @constraint(model1, θ >= cut.α + cut.β[1] * x1)
    end
    @objective(model1, Min, SU*u1 + F*x1 + c*g1 + M*shed1 + θ)
    optimize!(model1)
    x1_opt      = round(Int, value(x1))
    u1_opt      = round(Int, value(u1))
    g1_opt      = value(g1)
    shed1_opt   = value(shed1)
    stage1_cost = SU*u1_opt + F*x1_opt + c*g1_opt + M*shed1_opt
    (x1=x1_opt, stage1_cost=stage1_cost)
end

# Exact expected stage-2 cost over both scenarios
function stage2_cost(d2::Float64, x1_val::Int)
    model2 = Model(HiGHS.Optimizer); set_silent(model2)
    @variable(model2, x2, Bin); @variable(model2, u2, Bin)
    @variable(model2, g2 >= 0); @variable(model2, shed2 >= 0)
    @variable(model2, 0 <= z2 <= 1)
    JuMP.fix(z2, Float64(x1_val); force=true)
    @constraint(model2, g2 + shed2 == d2)
    @constraint(model2, g2 <= Gmax * x2)
    @constraint(model2, u2 >= x2 - z2)
    @objective(model2, Min, SU*u2 + F*x2 + c*g2 + M*shed2)
    optimize!(model2)
    return objective_value(model2)
end

x1_opt      = x1_opt_val.x1
stage1_cost = x1_opt_val.stage1_cost

costs_exact = [stage2_cost(d2, x1_opt) for d2 in scenarios]
ev_stage2   = dot(probs, costs_exact)
ev_total    = stage1_cost + ev_stage2

println()
println("  Exact expectation (enumeration over all scenarios):")
for (d2, p, cost) in zip(scenarios, probs, costs_exact)
    @printf "    d2 = %.1f MW  (p=%.2f):  stage-2 cost = %.4f\n" d2 p cost
end
@printf "  E[stage-2 cost]  = %.4f\n" ev_stage2
@printf "  Stage-1 op cost  = %.4f\n" stage1_cost
@printf "  E[total cost]    = %.4f\n" ev_total

# Monte Carlo estimate
println()
println("  Monte Carlo simulation (N = 10 000 rollouts):")
Random.seed!(42)
N      = 10_000
sample = [stage2_cost(rand() < 0.5 ? 4.0 : 9.0, x1_opt) for _ in 1:N]
mc_s2_mean = mean(sample)
mc_s2_std  = std(sample) / sqrt(N)
mc_total   = stage1_cost + mc_s2_mean
@printf "  E[stage-2 cost]  = %.4f  ± %.4f (95%% CI: ±%.4f)\n" mc_s2_mean mc_s2_std 1.96*mc_s2_std
@printf "  E[total cost]    = %.4f\n" mc_total

println()
println("="^60)
println("Done.")
println("="^60)
