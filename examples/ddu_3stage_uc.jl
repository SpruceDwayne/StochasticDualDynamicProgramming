using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
using SDDPBAPE
using JuMP, HiGHS
using LinearAlgebra

# ============================================================================
# 3-Stage DDU Unit Commitment Example
# ============================================================================
#
# A 3-stage binary unit-commitment problem with decision-dependent uncertainty.
#
# Generator parameters
const Gmax  = 10.0
const SU    = 2.0
const F     = 0.5
const c_g   = 1.0
const M_sh  = 100.0   # load-shed penalty
const M_big = 2000.0  # big-M for DDU cut activation
#
# Stage 1 (deterministic, d1=6): decide x1 ∈ {0,1}.
#   Region 1 (x1=0) → stage-2 demand {3.0, 4.0} with equal probability.
#   Region 2 (x1=1) → stage-2 demand {7.0, 8.0} with equal probability.
#   x_state = [] (no parent state to fix; x1 is a FREE decision variable).
#   x_next = [x1].
#
# Stage 2 (stochastic): decide x2 ∈ {0,1}, z2 copies x1.
#   Region 1 (x2=0) → stage-3 demand {2.0, 3.0} with equal probability.
#   Region 2 (x2=1) → stage-3 demand {6.0, 7.0} with equal probability.
#   x_state = [x2] (entering state), z_vars = [z2], x_next = [x2].
#
# Stage 3 (terminal): decide x3 ∈ {0,1}, z3 copies x2.
#   x_state = [x3], z_vars = [z3], x_next = [x3].
#   region_indicators = Dict() (terminal, no outgoing regions).
#   θ == 0.
#
# KEY DDU INDEXING DEMONSTRATION: x_leaving[s][2] is x2 (stage-2 OUTPUT),
# which differs from x_entering[s][2] = x1 (stage-2 INPUT) when stage 2 changes
# the generator state. The backward pass at step t=2 must use x_leaving, not x_entering.
#
# ============================================================================

# ── Region definitions ───────────────────────────────────────────────────────
# Stage-1 outgoing regions (indexed by x1 decision)
const REGION_1_S1 = DDURegion(1, Any[3.0, 4.0], [0.5, 0.5])  # x1=0 → low demand
const REGION_2_S1 = DDURegion(2, Any[7.0, 8.0], [0.5, 0.5])  # x1=1 → high demand

# Stage-2 outgoing regions (indexed by x2 decision)
const REGION_1_S2 = DDURegion(1, Any[2.0, 3.0], [0.5, 0.5])  # x2=0 → low demand
const REGION_2_S2 = DDURegion(2, Any[6.0, 7.0], [0.5, 0.5])  # x2=1 → high demand

# Root region for stage 1 (dummy, deterministic)
const REGION_0    = DDURegion(0, Any[nothing], [1.0])
const ζ_INIT_3S   = 0

# ── Stage builders ───────────────────────────────────────────────────────────

function build_stage1_3s(t, vf_next, ω; fix_state)
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, x1, Bin)
    @variable(model, u1, Bin)
    @variable(model, g1 >= 0); @variable(model, shed1 >= 0)
    @variable(model, θ >= -1e8)
    d1 = 6.0
    @constraint(model, g1 + shed1 == d1)
    @constraint(model, g1 <= Gmax * x1)
    @constraint(model, u1 >= x1)
    @variable(model, ind1, Bin); @variable(model, ind2, Bin)
    @constraint(model, ind1 + ind2 == 1)
    @constraint(model, ind2 == x1)
    @objective(model, Min, SU*u1 + F*x1 + c_g*g1 + M_sh*shed1 + θ)
    misc = Dict{Symbol,Any}(
        :x_next            => [x1],
        :region_indicators => Dict{Int,JuMP.VariableRef}(1 => ind1, 2 => ind2),
    )
    # x_state is empty: stage 1 has no parent state to fix.
    # x1 is a FREE decision variable, not an inherited state.
    return model, JuMP.VariableRef[], θ, misc
end

function build_stage2_3s(t, vf_next, ω; fix_state)
    d2 = ω
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, x2, Bin)
    @variable(model, u2, Bin)
    @variable(model, g2 >= 0); @variable(model, shed2 >= 0)
    @variable(model, 0 <= z2 <= 1)
    @constraint(model, z2 == fix_state[1])  # replaced by JuMP.fix
    @constraint(model, g2 + shed2 == d2)
    @constraint(model, g2 <= Gmax * x2)
    @constraint(model, u2 >= x2 - z2)
    @variable(model, θ >= -1e8)
    @variable(model, ind1, Bin); @variable(model, ind2, Bin)
    @constraint(model, ind1 + ind2 == 1)
    @constraint(model, ind2 == x2)
    @objective(model, Min, SU*u2 + F*x2 + c_g*g2 + M_sh*shed2 + θ)
    misc = Dict{Symbol,Any}(
        :z_vars            => [z2],
        :x_next            => [x2],
        :region_indicators => Dict{Int,JuMP.VariableRef}(1 => ind1, 2 => ind2),
    )
    return model, [x2], θ, misc
end

function build_stage3_3s(t, vf_next, ω; fix_state)
    d3 = ω
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, x3, Bin)
    @variable(model, u3, Bin)
    @variable(model, g3 >= 0); @variable(model, shed3 >= 0)
    @variable(model, 0 <= z3 <= 1)
    @constraint(model, z3 == fix_state[1])
    @constraint(model, g3 + shed3 == d3)
    @constraint(model, g3 <= Gmax * x3)
    @constraint(model, u3 >= x3 - z3)
    @variable(model, θ == 0)
    @objective(model, Min, SU*u3 + F*x3 + c_g*g3 + M_sh*shed3 + θ)
    misc = Dict{Symbol,Any}(
        :z_vars            => [z3],
        :x_next            => [x3],
        :region_indicators => Dict{Int,JuMP.VariableRef}(),
    )
    return model, [x3], θ, misc
end

# ── Stage assembly ───────────────────────────────────────────────────────────

stage1_3s = Stage(
    1, 1,
    build_stage1_3s,
    (ζ) -> nothing,                               # sampler: deterministic stage 1
    _ -> 1.0,
    (t, ζ) -> (REGION_0.Xi, REGION_0.pXi),       # children: unused by backward at t=1
    (t, ζ, ω) -> ζ,                               # next_ctx: no-op placeholder
    x -> x,
)

stage2_3s = Stage(
    2, 1,
    build_stage2_3s,
    (ζ) -> begin                                   # sampler: draw from μ_{ζ}
        reg = ζ == 1 ? REGION_1_S1 : REGION_2_S1
        idx = rand(1:length(reg.Xi))
        reg.Xi[idx]
    end,
    _ -> 1.0,
    (t, ζ) -> begin                                # children: full support of μ_{ζ} at stage 1
        reg = ζ == 1 ? REGION_1_S1 : REGION_2_S1
        (reg.Xi, reg.pXi)
    end,
    (t, ζ, ω) -> ζ,
    x -> x,
)

stage3_3s = Stage(
    3, 1,
    build_stage3_3s,
    (ζ) -> begin                                   # sampler: draw from μ_{ζ} at stage 2
        reg = ζ == 1 ? REGION_1_S2 : REGION_2_S2
        idx = rand(1:length(reg.Xi))
        reg.Xi[idx]
    end,
    _ -> 1.0,
    (t, ζ) -> begin                                # children: full support of μ_{ζ} at stage 2
        reg = ζ == 1 ? REGION_1_S2 : REGION_2_S2
        (reg.Xi, reg.pXi)
    end,
    (t, ζ, ω) -> ζ,
    x -> x,
)

# ── Region lists per stage ────────────────────────────────────────────────────
regions_s1 = [REGION_1_S1, REGION_2_S1]   # D_1 = {1, 2}
regions_s2 = [REGION_1_S2, REGION_2_S2]   # D_2 = {1, 2}
regions_s3 = DDURegion[]                   # D_3 = {} (terminal)

# ── Model assembly ────────────────────────────────────────────────────────────
m_3s = DDUSDDP(
    [stage1_3s, stage2_3s, stage3_3s],
    [regions_s1, regions_s2, regions_s3];
    M_big = M_big,
)

x0_3s = [0.0]
config_3s = SDDiPConfig(cut_type = :SB)

println("="^70)
println("3-Stage DDU Unit Commitment Example")
println("="^70)

# ── Training loop ─────────────────────────────────────────────────────────────
println("\nRunning DDU SDDiP for up to 50 iterations (patience=10, force_every=1) ...")
result = run_ddu_sddip!(m_3s;
    x0          = x0_3s,
    ζ_init      = ζ_INIT_3S,
    config      = config_3s,
    S           = 1,
    max_iter    = 50,
    patience    = 10,
    force_every = 1,
    cut_atol    = 0.0,
    evaluate_stage = 1,
    evaluate_δ     = 1,
)

println("\nConverged after $(result.iters) iterations.")
println("Cuts per stage: $(result.cuts_per_stage)")

# ── Print value function summaries ────────────────────────────────────────────
println("\n--- Value function V[1] (stage-1 outgoing regions) ---")
for d in [1, 2]
    if haskey(m_3s.V[1], d)
        vf = m_3s.V[1][d]
        val, _ = evaluate(vf, x0_3s)
        println("  V[1][$d] has $(length(vf.cuts)) cuts; V[1][$d](x0=$(x0_3s)) = $val")
    else
        println("  V[1][$d] — no cuts generated")
    end
end

println("\n--- Value function V[2] (stage-2 outgoing regions) ---")
for d in [1, 2]
    if haskey(m_3s.V[2], d)
        vf = m_3s.V[2][d]
        val0, _ = evaluate(vf, [0.0])
        val1, _ = evaluate(vf, [1.0])
        println("  V[2][$d] has $(length(vf.cuts)) cuts; at x2=0: $val0, at x2=1: $val1")
    else
        println("  V[2][$d] — no cuts generated")
    end
end

# ── Forward pass demonstration ────────────────────────────────────────────────
println("\n--- Forward pass demonstration (x0=[0.0]) ---")
# Run a single forward pass to inspect the decisions
m_demo = DDUSDDP(
    [stage1_3s, stage2_3s, stage3_3s],
    [regions_s1, regions_s2, regions_s3];
    M_big = M_big,
)
# Copy the trained cuts into m_demo
for t in 1:3
    for (d, vf) in m_3s.V[t]
        vf_demo = get_V_ddu!(m_demo, t, d)
        for cut in vf.cuts
            add_cut!(vf_demo, cut.α, cut.β, cut.stage)
        end
    end
end
fwd_demo = forward_pass_ddu_online!(m_demo; S=1, x0=x0_3s, ζ_init=ζ_INIT_3S)

s = 1
x1_demo = fwd_demo.x_leaving[s][1][1]
x2_entering = fwd_demo.x_entering[s][2][1]  # = x1, the state entering stage 2
x2_leaving  = fwd_demo.x_leaving[s][2][1]   # = x2, the decision made at stage 2
x3_entering = fwd_demo.x_entering[s][3][1]  # = x2, the state entering stage 3
δ1 = fwd_demo.δ_hist[s][1]
δ2 = fwd_demo.δ_hist[s][2]

println("  x_leaving[s][1]  = x1 = $x1_demo  (stage-1 decision, generator on=1/off=0)")
println("  x_entering[s][2] = x1 = $x2_entering  (stage-2 INPUT, inherited from stage 1)")
println("  x_leaving[s][2]  = x2 = $x2_leaving  (stage-2 OUTPUT, the new generator state)")
println("  x_entering[s][3] = x2 = $x3_entering  (stage-3 INPUT)")
println("  δ_hist[s][1] = $δ1  (outgoing region at stage 1)")
println("  δ_hist[s][2] = $δ2  (outgoing region at stage 2)")

if x2_entering != x2_leaving
    println("\n  KEY RESULT: x_leaving[s][2] ($x2_leaving) ≠ x_entering[s][2] ($x2_entering)")
    println("  Stage 2 changed the generator state — demonstrating that x_leaving ≠ x_entering")
    println("  at intermediate stages. The backward pass at step t=2 correctly uses x_leaving.")
else
    println("\n  NOTE: x_leaving[s][2] == x_entering[s][2] in this run (generator held its state).")
    println("  This is valid; the key DDU indexing distinction still applies structurally.")
end

# ── Assertions ────────────────────────────────────────────────────────────────
println("\n--- Assertions ---")

@assert result.iters >= 1  "should run at least 1 iteration"
@assert sum(result.cuts_per_stage) >= 1  "should generate at least 1 cut total"

# V[1] should have at least one non-empty region
n_v1_nonempty = count(d -> haskey(m_3s.V[1], d.id) && !isempty(m_3s.V[1][d.id].cuts),
                      regions_s1)
@assert n_v1_nonempty >= 1  "V[1] should have at least one non-empty region"

# x_leaving[s][1] should be binary (0 or 1)
@assert x1_demo ≈ 0.0 || x1_demo ≈ 1.0  "x_leaving[s][1] should be binary (0 or 1), got $x1_demo"

println("  PASS: result.iters >= 1")
println("  PASS: sum(cuts_per_stage) >= 1")
println("  PASS: V[1] has at least $n_v1_nonempty non-empty region(s)")
println("  PASS: x_leaving[s][1] = $x1_demo is binary")

println("\n" * "="^70)
println("3-STAGE DDU EXAMPLE COMPLETE")
println("="^70)
