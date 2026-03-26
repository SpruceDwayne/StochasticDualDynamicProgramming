using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
using SDDPBAPE
using JuMP, HiGHS, LinearAlgebra, Printf, Random
# ============================================================================
# Problem: 3-Stage Stochastic Binary Unit Commitment
# ============================================================================
#
# Same model as the 2-stage version, extended by one period.
#
# Decision variables per stage t ∈ {1, 2, 3}:
#   x_t  ∈ {0,1}   generator ON/OFF (also the binary state variable)
#   u_t  ∈ {0,1}   startup indicator (1 if turning on this period)
#   g_t  ≥ 0       energy generated [MWh]
#   shed_t ≥ 0     unmet demand [MWh]
#
# Per-stage cost:  SU·u_t + F·x_t + c·g_t + M·shed_t
#
# Constraints:
#   g_t + shed_t = d_t            power balance
#   g_t ≤ Gmax · x_t             capacity
#   u_t ≥ x_t − x_{t-1}         startup logic
#
# SDDiP copy constraint (stages 2 and 3):
#   z_t ∈ [0,1] copies the parent binary state x_{t-1}.
#   Relaxing z_t = x_{t-1} via multiplier π yields Lagrangian cuts
#     θ ≥ L(π*) + π*·x_{t-1}
#   which are added to the stage-(t-1) value-function approximation V[t].
#
# Stochasticity:
#   d_1 = 6.0 (deterministic)
#   d_2 ∈ {4.0, 9.0} with p = 0.5 each
#   d_3 ∈ {4.0, 9.0} with p = 0.5 each  (independent of d_2)
#
# True optima (by backward induction):
#   V[3](x2=0) = 9.0,  V[3](x2=1) = 7.0
#   V[2](x1=0) = 16.0, V[2](x1=1) = 14.0  → cut: θ ≥ 16 − 2·x1
#   Optimal x1 = 1 (ON);  E[total cost] = 8.5 + 14.0 = 22.5
#
# Note on V[3]: x2=0 is never reached in the forward pass (shedding is
# far more expensive than running the generator), so the Lagrangian cut at
# z3=1 has LP-dual = 0 and produces only α=7, β=0 (a valid lower bound but
# not tight at x2=0).  V[2] and the overall cost are still exact because the
# stage-2 Lagrangian captures the full 2-stage-3 cost correctly.
# ============================================================================

const Gmax = 10.0
const d1   = 6.0
const SU   = 2.0
const F    = 0.5
const c    = 1.0
const M    = 100.0

const scenarios = [4.0, 9.0]
const probs     = [0.5, 0.5]

# ── Stage builders ────────────────────────────────────────────────────────────

# Stage 1: deterministic, no copy variable (no parent binary state).
function build_stage1(t, vf_next, ω; fix_state)
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, x1, Bin)
    @variable(model, u1, Bin)
    @variable(model, g1    >= 0)
    @variable(model, shed1 >= 0)
    @variable(model, θ     >= 0)
    @constraint(model, g1 + shed1 == d1)
    @constraint(model, g1 <= Gmax * x1)
    @constraint(model, u1 >= x1)                  # start from OFF
    for cut in vf_next.cuts
        @constraint(model, θ >= cut.α + cut.β[1] * x1)
    end
    @objective(model, Min, SU*u1 + F*x1 + c*g1 + M*shed1 + θ)
    misc = Dict{Symbol,Any}(:x_next => [x1], :x_state => [x1])
    return model, [x1], θ, misc
end

# Stage 2: intermediate — copy variable z2 copies x1; θ approximates E[V[3](x2)].
function build_stage2(t, vf_next, ω; fix_state)
    d2 = ω
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, x2, Bin)
    @variable(model, u2, Bin)
    @variable(model, g2    >= 0)
    @variable(model, shed2 >= 0)
    @variable(model, 0 <= z2 <= 1)          # copies parent x1
    @constraint(model, z2 == fix_state[1])
    @constraint(model, g2 + shed2 == d2)
    @constraint(model, g2 <= Gmax * x2)
    @constraint(model, u2 >= x2 - z2)
    @variable(model, θ >= 0)                 # cost-to-go for V[3]
    for cut in vf_next.cuts
        @constraint(model, θ >= cut.α + cut.β[1] * x2)
    end
    @objective(model, Min, SU*u2 + F*x2 + c*g2 + M*shed2 + θ)
    misc = Dict{Symbol,Any}(:z_vars => [z2], :x_next => [x2], :x_state => [x2])
    return model, [x2], θ, misc
end

# Stage 3: terminal — copy variable z3 copies x2; θ = 0.
function build_stage3(t, vf_next, ω; fix_state)
    d3 = ω
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, x3, Bin)
    @variable(model, u3, Bin)
    @variable(model, g3    >= 0)
    @variable(model, shed3 >= 0)
    @variable(model, 0 <= z3 <= 1)          # copies parent x2
    @constraint(model, z3 == fix_state[1])
    @constraint(model, g3 + shed3 == d3)
    @constraint(model, g3 <= Gmax * x3)
    @constraint(model, u3 >= x3 - z3)
    @variable(model, θ == 0)                 # terminal
    @objective(model, Min, SU*u3 + F*x3 + c*g3 + M*shed3 + θ)
    misc = Dict{Symbol,Any}(:z_vars => [z3], :x_next => [x3], :x_state => [x3])
    return model, [x3], θ, misc
end

# ── Assemble model ────────────────────────────────────────────────────────────
stage1 = Stage(1, 1, build_stage1,
    () -> nothing,
    _ -> 1.0,
    (t, ctx) -> ([nothing], [1.0]),
    (t, ctx, ω) -> nothing,
    x -> x,
)
stage2 = Stage(2, 1, build_stage2,
    () -> scenarios[rand(1:2)],
    _ -> 1.0,
    (t, ctx) -> (scenarios, probs),
    (t, ctx, ω) -> nothing,
    x -> x,
)
stage3 = Stage(3, 1, build_stage3,
    () -> scenarios[rand(1:2)],
    _ -> 1.0,
    (t, ctx) -> (scenarios, probs),
    (t, ctx, ω) -> nothing,
    x -> x,
)

m  = SDDP([stage1, stage2, stage3])
x0 = [0.0]   # initial state: generator OFF

# ── Train SDDiP ───────────────────────────────────────────────────────────────
config = SDDiPConfig(
    cut_type       = :lagrangian,
    lag_tol        = 1e-4,
    lag_max_iter   = 200,
    step_size_init = 1.0,
    step_decay     = 0.95,
)

println("="^60)
println("3-Stage Binary Unit Commitment  —  SDDiP (Lagrangian cuts)")
println("="^60)
println()

result = run_sddip!(
    m;
    x0             = x0,
    config         = config,
    max_iter       = 200,
    patience       = 10,
    force_every    = 5,
    evaluate_index = 2,
    cut_atol       = 1e-8,
)

println()
@printf "Converged in %d iterations.  Cuts: V[2]=%d, V[3]=%d\n" result.iters result.cuts_per_stage[2] result.cuts_per_stage[3]

# ── Value functions ───────────────────────────────────────────────────────────
println()
println("─"^60)
# V[2]: cuts for stage-1 epigraph  (tight at both x1=0 and x1=1)
println("V[2](x1) — used in stage-1 epigraph:")
@printf "  %5s  %8s  %8s\n" "cut" "α" "β"
for (k, cut) in enumerate(m.V[2].cuts)
    @printf "  %5d  %8.4f  %8.4f\n" k cut.α cut.β[1]
end
v2_0, _ = evaluate(m.V[2], [0.0])
v2_1, _ = evaluate(m.V[2], [1.0])
@printf "  V[2](0)=%.4f  (true 16.0)    V[2](1)=%.4f  (true 14.0)\n" v2_0 v2_1
println()

# V[3]: cuts for stage-2 epigraph  (tight only at visited state x2=1;
# x2=0 is never reached so V[3](0) underestimates the true 9.0)
println("V[3](x2) — used in stage-2 epigraph  [tight at x2=1 only]:")
@printf "  %5s  %8s  %8s\n" "cut" "α" "β"
for (k, cut) in enumerate(m.V[3].cuts)
    @printf "  %5d  %8.4f  %8.4f\n" k cut.α cut.β[1]
end
v3_1, _ = evaluate(m.V[3], [1.0])
@printf "  V[3](1)=%.4f  (true 7.0)\n" v3_1
println()

# ── Optimal stage-1 decision ──────────────────────────────────────────────────
println("─"^60)
println("Optimal stage-1 decisions  (starting from OFF)")
println("─"^60)
x1_opt, stage1_cost = let m1 = Model(HiGHS.Optimizer)
    set_silent(m1)
    @variable(m1, x1, Bin); @variable(m1, u1, Bin)
    @variable(m1, g1 >= 0); @variable(m1, shed1 >= 0)
    @variable(m1, θ >= 0)
    @constraint(m1, g1 + shed1 == d1)
    @constraint(m1, g1 <= Gmax * x1)
    @constraint(m1, u1 >= x1)
    for cut in m.V[2].cuts
        @constraint(m1, θ >= cut.α + cut.β[1] * x1)
    end
    @objective(m1, Min, SU*u1 + F*x1 + c*g1 + M*shed1 + θ)
    optimize!(m1)
    x1v = round(Int, value(x1)); u1v = round(Int, value(u1))
    g1v = value(g1); shed1v = value(shed1)
    c1  = SU*u1v + F*x1v + c*g1v + M*shed1v
    @printf "  x1=%d  u1=%d  g1=%.2f  shed1=%.2f\n" x1v u1v g1v shed1v
    @printf "  Stage-1 op cost : %.4f\n" c1
    @printf "  Continuation θ  : %.4f  [≈ E[V[2](x1)]]\n" value(θ)
    @printf "  Stage-1 total   : %.4f\n" objective_value(m1)
    (x1v, c1)
end

# ── Expected total cost (exact enumeration over 4 scenario paths) ─────────────
println()
println("─"^60)
println("Expected total cost  (exact: enumerate all 4 paths)")
println("─"^60)
println()

# Solve stage 2 and 3 greedily (turning ON is always dominant for these params).
function stage23_cost(d2::Float64, d3::Float64, x1_val::Int)
    # Stage 2
    m2 = Model(HiGHS.Optimizer); set_silent(m2)
    @variable(m2, x2, Bin); @variable(m2, u2, Bin)
    @variable(m2, g2 >= 0); @variable(m2, shed2 >= 0)
    @variable(m2, 0 <= z2 <= 1); JuMP.fix(z2, Float64(x1_val); force=true)
    @constraint(m2, g2 + shed2 == d2); @constraint(m2, g2 <= Gmax * x2)
    @constraint(m2, u2 >= x2 - z2)
    @objective(m2, Min, SU*u2 + F*x2 + c*g2 + M*shed2)
    optimize!(m2)
    x2_opt = round(Int, value(x2))
    c2     = objective_value(m2)

    # Stage 3
    m3 = Model(HiGHS.Optimizer); set_silent(m3)
    @variable(m3, x3, Bin); @variable(m3, u3, Bin)
    @variable(m3, g3 >= 0); @variable(m3, shed3 >= 0)
    @variable(m3, 0 <= z3 <= 1); JuMP.fix(z3, Float64(x2_opt); force=true)
    @constraint(m3, g3 + shed3 == d3); @constraint(m3, g3 <= Gmax * x3)
    @constraint(m3, u3 >= x3 - z3)
    @objective(m3, Min, SU*u3 + F*x3 + c*g3 + M*shed3)
    optimize!(m3)
    c3 = objective_value(m3)
    return c2, c3
end

let ev_23 = 0.0
    for (d2, p2) in zip(scenarios, probs), (d3, p3) in zip(scenarios, probs)
        c2, c3  = stage23_cost(d2, d3, x1_opt)
        p_path   = p2 * p3
        ev_23   += p_path * (c2 + c3)
        @printf "  d2=%.0f d3=%.0f (p=%.2f):  c2=%.4f  c3=%.4f\n" d2 d3 p_path c2 c3
    end
    println()
    @printf "  E[c2+c3]       = %.4f\n" ev_23
    @printf "  Stage-1 cost   = %.4f\n" stage1_cost
    @printf "  E[total cost]  = %.4f\n" stage1_cost + ev_23
end

println()
println("="^60)
println("Done.")
println("="^60)
