using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
using SDDPBAPE
using JuMP, HiGHS
using LinearAlgebra
using Random
using Printf

###########################
#Note this script is corrupted. Do not use it 
###########################


# ============================================================================
# Trading problem: DDU variant with 50 shares and 4 computational stages
#
# Stages (Julia 1–4, math t=0–3):
#   Stage 1: root (deterministic b₀=1); buy xO₁ ∈ {0,...,50}.
#   Stage 2: free decision; buy xO₂ ∈ {0,...,xP₁}; cost = b·xO₂ + m·xO₁.
#   Stage 3: forced buy xP₂ (all remaining);       cost = b·xP₂ + m·xO₂.
#   Stage 4: terminal markup;                       cost = m·xP₂.
#
# Every purchase eventually pays its markup one stage later (same as the full
# 5-stage/100-share problem). Spreading purchases is optimal because concentrating
# a large order at any stage puts it in a high-markup region, which inflates the
# cost at the very next stage.
# ============================================================================

const N_SHARES  = 50
const N_BITS    = 6          # 2^6 = 64 > 50
const STATE_DIM = 2 * N_BITS # 12 bits: 6 for xP, 6 for xO_prev

const B_VALS  =[1.00,1.00] #[0.9985, 1.00, 1.0015, 1.003]
const B_PROBS = fill(0.5,2) #fill(0.2, 5)

const M_VALS  = [0.01, 0.08, 0.118]
const M_PROBS = [0.85 0.15 0.00;
                 0.60 0.35 0.05;
                 0.25 0.25 0.50;
                 0.10 0.10 0.80]

# Region bounds scaled proportionally from 100-share to 50-share problem
const REG_L = [0,  11, 21, 41]
const REG_U = [10, 20, 40, 50]

const BIN_COEFFS = [2.0^(i-1) for i in 1:N_BITS]

# ─── DDU Regions ─────────────────────────────────────────────────────────────
function make_joint_region(d::Int)
    mp = M_PROBS[d, :]
    Xi, pXi = Any[], Float64[]
    for (b, pb) in zip(B_VALS, B_PROBS)
        for k in 1:3
            push!(Xi,  (b, M_VALS[k]))
            push!(pXi, pb * mp[k])
        end
    end
    @assert abs(sum(pXi) - 1.0) < 1e-10
    return DDURegion(d, Xi, pXi)
end

const TRADING_REGIONS = [make_joint_region(d) for d in 1:4]
const REGION_0        = DDURegion(0, Any[(1.0, 0.0)], [1.0])
const ζ_INIT          = 0

const REGION_MAP = Dict{Int, DDURegion}(
    0 => REGION_0,
    [d => TRADING_REGIONS[d] for d in 1:4]...
)

# ─── Helpers ─────────────────────────────────────────────────────────────────
function int_to_bits(v::Int)::Vector{Float64}
    @assert 0 <= v <= N_SHARES "value $v out of range [0, $N_SHARES]"
    return Float64[Float64((v >> (i-1)) & 1) for i in 1:N_BITS]
end

const X0 = vcat(int_to_bits(N_SHARES), int_to_bits(0))

function add_region_constraints!(model, xO_var, δ)
    @constraint(model, sum(δ) == 1)
    for d in 1:4
        @constraint(model, xO_var >= REG_L[d] - N_SHARES * (1 - δ[d]))
        @constraint(model, xO_var <= REG_U[d] + N_SHARES * (1 - δ[d]))
    end
end

function add_z_vars!(model, fix_state)
    n = STATE_DIM
    @variable(model, 0 <= z[1:n] <= 1)
    for i in 1:n
        JuMP.fix(z[i], fix_state[i]; force = true)
    end
    xP_in = sum(BIN_COEFFS[i] * z[i]          for i in 1:N_BITS)
    xO_in = sum(BIN_COEFFS[i] * z[N_BITS + i] for i in 1:N_BITS)
    return collect(JuMP.VariableRef, z), xP_in, xO_in
end

function add_binary_state!(model, xP_next_expr, xO_next_expr)
    @variable(model, bP[1:N_BITS], Bin)
    @variable(model, bO[1:N_BITS], Bin)
    @constraint(model, dot(BIN_COEFFS, bP) == xP_next_expr)
    @constraint(model, dot(BIN_COEFFS, bO) == xO_next_expr)
    return JuMP.VariableRef[bP; bO]
end

function dp_region(xO::Int)::Int
    for d in 1:4
        REG_L[d] <= xO <= REG_U[d] && return d
    end
    error("xO=$xO out of [0,$N_SHARES]")
end

# ============================================================================
# Stage builders
# ============================================================================

# ── Stage 1: root, deterministic (b₀=1, no markup since xO_prev=0) ───────────
function build_stage1(t, vf_next, ω; fix_state)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, 0 <= xO <= N_SHARES)
    x_next = add_binary_state!(model, N_SHARES - xO, xO)

    @variable(model, δ[1:4], Bin)
    add_region_constraints!(model, xO, δ)

    @variable(model, θ >= 0)
    @objective(model, Min, 1.0 * xO + θ)

    misc = Dict{Symbol, Any}(
        :x_next            => x_next,
        :region_indicators => Dict{Int, JuMP.VariableRef}(d => δ[d] for d in 1:4),
    )
    return model, JuMP.VariableRef[], θ, misc
end

# ── Stage 2: free trading decision ───────────────────────────────────────────
function build_stage2(t, vf_next, ω; fix_state)
    b, m = ω
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    z_vars, xP_in, xO_in = add_z_vars!(model, fix_state)

    @variable(model, 0 <= xO <= N_SHARES)
    @constraint(model, xO <= xP_in)
    x_next = add_binary_state!(model, xP_in - xO, xO)

    @variable(model, δ[1:4], Bin)
    add_region_constraints!(model, xO, δ)

    @variable(model, θ >= 0)
    @objective(model, Min, b * xO + m * xO_in + θ)

    misc = Dict{Symbol, Any}(
        :z_vars            => z_vars,
        :x_next            => x_next,
        :region_indicators => Dict{Int, JuMP.VariableRef}(d => δ[d] for d in 1:4),
    )
    return model, JuMP.VariableRef[], θ, misc
end

# ── Stage 3: forced buy all remaining shares ──────────────────────────────────
# Pays markup on xO₂ (stage 2's order). Its own forced buy xP₂ will pay markup
# at stage 4. Region indicators link xP₂ to stage 4's markup distribution.
function build_stage3(t, vf_next, ω; fix_state)
    b, m = ω
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    z_vars, xP_in, xO_in = add_z_vars!(model, fix_state)

    @variable(model, 0 <= xO <= N_SHARES)
    @constraint(model, xO == xP_in)   # forced: must buy all remaining

    x_next = add_binary_state!(model, 0, xO)   # xP_next=0, xO_next=forced buy amount

    @variable(model, δ[1:4], Bin)
    add_region_constraints!(model, xO, δ)      # region of forced buy → stage 4 markup

    @variable(model, θ >= 0)
    @objective(model, Min, b * xO + m * xO_in + θ)

    misc = Dict{Symbol, Any}(
        :z_vars            => z_vars,
        :x_next            => x_next,
        :region_indicators => Dict{Int, JuMP.VariableRef}(d => δ[d] for d in 1:4),
    )
    return model, JuMP.VariableRef[], θ, misc
end

# ── Stage 4: terminal — pay markup on stage 3's forced buy (xO_in = xP₂) ─────
function build_stage4(t, vf_next, ω; fix_state)
    b, m = ω   # b unused (no new purchase)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    z_vars, _xP_in, xO_in = add_z_vars!(model, fix_state)

    @variable(model, θ == 0)   # terminal
    @objective(model, Min, m * xO_in + θ)

    misc = Dict{Symbol, Any}(
        :z_vars            => z_vars,
        :x_next            => JuMP.VariableRef[],
        :region_indicators => Dict{Int, JuMP.VariableRef}(),
    )
    return model, JuMP.VariableRef[], θ, misc
end

# ============================================================================
# Stage objects and DDU model
# ============================================================================

function make_sampler_fn(region_map)
    return function(ζ::Int)
        reg = region_map[ζ]
        r, cum = rand(), 0.0
        for (xi, pi) in zip(reg.Xi, reg.pXi)
            cum += pi
            cum >= r && return xi
        end
        return last(reg.Xi)
    end
end

function make_children_fn(region_map)
    return function(t::Int, ζ::Int)
        reg = region_map[ζ]
        return (reg.Xi, reg.pXi)
    end
end

const SAMPLER  = make_sampler_fn(REGION_MAP)
const CHILDREN = make_children_fn(REGION_MAP)

stage1 = Stage(1, STATE_DIM, build_stage1, SAMPLER, _ -> 1.0, CHILDREN, (t,ζ,ω) -> ζ, x -> x)
stage2 = Stage(2, STATE_DIM, build_stage2, SAMPLER, _ -> 1.0, CHILDREN, (t,ζ,ω) -> ζ, x -> x)
stage3 = Stage(3, STATE_DIM, build_stage3, SAMPLER, _ -> 1.0, CHILDREN, (t,ζ,ω) -> ζ, x -> x)
stage4 = Stage(4, STATE_DIM, build_stage4, SAMPLER, _ -> 1.0, CHILDREN, (t,ζ,ω) -> ζ, x -> x)

const REGIONS_PER_STAGE = [
    TRADING_REGIONS,   # stage 1: region(xO₁) → stage 2 markup distribution
    TRADING_REGIONS,   # stage 2: region(xO₂) → stage 3 markup distribution
    TRADING_REGIONS,   # stage 3: region(xP₂) → stage 4 markup distribution
    DDURegion[],       # stage 4: terminal
]

# Max cost: 50 shares × (max_b + max_m) per active stage × 4 stages ≈ 400.
const M_BIG = 1000.0

model_ddu = DDUSDDP(
    [stage1, stage2, stage3, stage4],
    REGIONS_PER_STAGE;
    M_big    = M_BIG,
    discount = 1.0,
)

config = SDDiPConfig(
    cut_type        = :lagrangian,
    burnin_iters    = 1,
    burnin_cut_type = :SB,
    level_cfg       = LevelMethodConfig(optimizer = HiGHS.Optimizer)
)

println("="^70)
println("DDU Trading Example: 50 shares, 4 stages")
println("="^70)
println("Stages: 4  |  Shares: $N_SHARES  |  Regions: 4  |  State bits: $STATE_DIM")
println("M_big = $M_BIG  |  Cut type: $(config.cut_type)")
println()

Random.seed!(42)
result = run_ddu_sddip!(model_ddu;
    x0          = X0,
    ζ_init      = ζ_INIT,
    config      = config,
    S           = 1,
    max_iter    = 500,
    patience    = 50,
    force_every = 10,
    cut_atol    = 1e-9,
)

println()
println("─── DDU-SDDiP result ───────────────────────────────────────────────")
println("Iterations : $(result.iters)")
println("Cuts/stage : $(result.cuts_per_stage)")
println("Total cuts : $(sum(result.cuts_per_stage))")

# ============================================================================
# Exact backward DP
# ============================================================================

println()
println("─── Exact backward DP ──────────────────────────────────────────────")

E_m = [dot(M_PROBS[d, :], M_VALS) for d in 1:4]
E_b = dot(B_PROBS, B_VALS)
println(@sprintf("E[b]              = %.4f", E_b))
println("E_d[m] per region = " * join([@sprintf("%.5f", e) for e in E_m], "  "))

# Stage 4 (terminal): pay markup on xO_prev (= xP₂, the stage-3 forced buy).
# V4[d][xO+1] = E_m[d] * xO   (xP = 0 always at stage 4)
V4 = [zeros(Float64, N_SHARES+1) for _ in 0:4]
for d in 1:4
    for xO in 0:N_SHARES
        V4[d+1][xO+1] = E_m[d] * xO
    end
end

# Stage 3 (forced buy xP, pay markup on xO_prev):
# V3[d][xP+1, xO+1] = E_{(b,m)~μ_d}[b·xP + m·xO + V4[d(xP)][xP+1]]
#                    = E_b·xP + E_m[d]·xO + E_m[d(xP)]·xP
#                    = (E_b + E_m[dp_region(xP)])·xP + E_m[d]·xO
V3 = [zeros(Float64, N_SHARES+1, N_SHARES+1) for _ in 0:4]
for d in 1:4
    for xP in 0:N_SHARES, xO in 0:N_SHARES
        xP + xO > N_SHARES && continue
        d_forced = dp_region(xP)
        V3[d+1][xP+1, xO+1] = (E_b + E_m[d_forced]) * xP + E_m[d] * xO
    end
end
println("Completed DP stage t=3 (forced buy, analytic)")

# Stage 2 (free decision): optimise over xO₂ ∈ {0,...,xP}
# V2[d][xP+1, xO+1] = E_{(b,m)~μ_d}[min_{xO₂} { b·xO₂ + m·xO + V3[d(xO₂)][xP-xO₂+1, xO₂+1] }]
V2 = [zeros(Float64, N_SHARES+1, N_SHARES+1) for _ in 0:4]
for d in 1:4
    reg = REGION_MAP[d]
    for xP in 0:N_SHARES, xO in 0:N_SHARES
        xP + xO > N_SHARES && continue
        total = 0.0
        for (scenario, prob) in zip(reg.Xi, reg.pXi)
            b_val, m_val = scenario
            markup = xO * m_val
            best = Inf
            for xO2 in 0:xP
                d_next = dp_region(xO2)
                val = xO2 * b_val + markup + V3[d_next+1][xP-xO2+1, xO2+1]
                val < best && (best = val)
            end
            total += prob * best
        end
        V2[d+1][xP+1, xO+1] = total
    end
end
println("Completed DP stage t=2 (free decision)")

# Stage 1 (root, deterministic): b₀=1, xO_prev=0, no markup at this stage
dp_costs  = [xO1 * 1.0 + V2[dp_region(xO1)+1][N_SHARES-xO1+1, xO1+1] for xO1 in 0:N_SHARES]
dp_optimal = minimum(dp_costs)
dp_opt_xO1 = argmin(dp_costs) - 1
println("Completed DP stage t=1 (root)")

lb = compute_ddu_lb!(model_ddu, X0, ζ_INIT)

println(@sprintf("Exact DP optimal cost : %.6f  (opt xO₁ = %d shares)", dp_optimal, dp_opt_xO1))
println(@sprintf("DDU-SDDiP lower bound : %.6f", lb))
println(@sprintf("Gap (LB vs exact)     : %.6f  (%.3f%%)",
    dp_optimal - lb, 100 * (dp_optimal - lb) / max(1e-10, abs(dp_optimal))))

# ============================================================================
# Policy simulation (Monte Carlo, exact DP policy)
# ============================================================================

println()
println("─── Policy simulation (Monte Carlo, exact DP policy) ───────────────")

N_SIM = 5000
Random.seed!(123)
sim_costs = zeros(Float64, N_SIM)

for s in 1:N_SIM
    xP, xO_prev, ζ = N_SHARES, 0, 0
    total = 0.0

    for stage_idx in 1:4
        reg   = REGION_MAP[ζ]
        r, cum = rand(), 0.0
        b_val, m_val = last(reg.Xi)
        for (xi, pi) in zip(reg.Xi, reg.pXi)
            cum += pi
            if r <= cum; b_val, m_val = xi; break; end
        end

        if stage_idx == 1
            # Root: deterministic; choose xO₁ to minimise total expected cost
            best_val, best_xO1 = Inf, 0
            for xO1 in 0:xP
                val = xO1 * 1.0 + V2[dp_region(xO1)+1][xP-xO1+1, xO1+1]
                if val < best_val; best_val = val; best_xO1 = xO1; end
            end
            total       += best_xO1 * 1.0
            ζ            = dp_region(best_xO1)
            xP, xO_prev  = xP - best_xO1, best_xO1

        elseif stage_idx == 2
            # Free decision: choose xO₂ given revealed (b, m)
            markup = xO_prev * m_val
            best_val, best_xO2 = Inf, 0
            for xO2 in 0:xP
                d_next = dp_region(xO2)
                val = xO2 * b_val + markup + V3[d_next+1][xP-xO2+1, xO2+1]
                if val < best_val; best_val = val; best_xO2 = xO2; end
            end
            total       += best_xO2 * b_val + markup
            ζ            = dp_region(best_xO2)
            xP, xO_prev  = xP - best_xO2, best_xO2

        elseif stage_idx == 3
            # Forced buy all remaining
            total       += xP * b_val + xO_prev * m_val
            ζ            = dp_region(xP)
            xP, xO_prev  = 0, xP

        else  # stage 4: terminal markup on stage-3's forced buy
            total += xO_prev * m_val
        end
    end
    sim_costs[s] = total
end

using Statistics
mc_mean = mean(sim_costs)
mc_std  = std(sim_costs)
mc_ci   = 1.96 * mc_std / sqrt(N_SIM)

println(@sprintf("MC mean (exact policy) : %.6f  ±  %.6f  (95%% CI, n=%d)", mc_mean, mc_ci, N_SIM))
println(@sprintf("Exact DP optimum       : %.6f", dp_optimal))
println(@sprintf("MC vs DP (should ≈ 0)  : %.6f  (%.4f%%)",
    mc_mean - dp_optimal, 100*abs(mc_mean - dp_optimal)/max(1e-10, abs(dp_optimal))))

println()
println("─── Final summary ──────────────────────────────────────────────────")
println(@sprintf("  DDU-SDDiP lower bound : %.6f", lb))
println(@sprintf("  Exact DP optimum      : %.6f  (xO₁* = %d shares)", dp_optimal, dp_opt_xO1))
println(@sprintf("  MC simulation (exact) : %.6f  ±  %.6f", mc_mean, mc_ci))
println(@sprintf("  Optimality gap        : %.4f%%",
    100 * (dp_optimal - lb) / max(1e-10, abs(dp_optimal))))

# ============================================================================
# Plots: recourse function and total expected cost vs first-stage order xO₁
# ============================================================================

println()
println("─── Computing cost-to-go curves ────────────────────────────────────")

# Direct evaluate(vf, x_state) extrapolates cuts far outside visited states,
# giving wildly pessimistic values (e.g. -M_BIG) for xO₁ values the algorithm
# never explored.  Instead, we fix the stage-1 output encoding in the actual
# stage-1 model and solve it — this applies the θ ≥ 0 lower bound and all
# big-M cut constraints, giving the proper SDDP lower bound for each xO₁.
ω_root    = REGION_0.Xi[1]   # (1.0, 0.0) — deterministic root scenario
model_s1, _, _, misc_s1 = get_or_build_ddu_model!(model_ddu, 1, ω_root, X0)
x_next_s1 = misc_s1[:x_next]   # [bP[1:N_BITS]; bO[1:N_BITS]]

n_orders    = collect(0:N_SHARES)
sddp_total  = Vector{Float64}(undef, N_SHARES + 1)
sddp_future = Vector{Float64}(undef, N_SHARES + 1)
dp_future   = Vector{Float64}(undef, N_SHARES + 1)
dp_total    = dp_costs

for xO1 in 0:N_SHARES
    # Fix output encoding → forces xO = xO1 via the binary encoding constraints
    target_bits = vcat(int_to_bits(N_SHARES - xO1), int_to_bits(xO1))
    for i in eachindex(x_next_s1)
        JuMP.fix(x_next_s1[i], target_bits[i]; force = true)
    end
    JuMP.optimize!(model_s1)
    obj = JuMP.objective_value(model_s1)
    sddp_total[xO1 + 1]  = obj
    sddp_future[xO1 + 1] = obj - xO1 * 1.0

    # DP future cost at this state (exact reference)
    d = dp_region(xO1)
    dp_future[xO1 + 1] = V2[d + 1][N_SHARES - xO1 + 1, xO1 + 1]
end

# Restore x_next_s1 to free binary variables for any subsequent model use
for var in x_next_s1
    JuMP.unfix(var)
    JuMP.set_lower_bound(var, 0.0)
    JuMP.set_upper_bound(var, 1.0)
    JuMP.set_binary(var)
end

sddp_opt_xO1 = argmin(sddp_total) - 1

println(@sprintf("  DP   optimal xO₁ : %d shares  (total cost %.6f)", dp_opt_xO1, dp_optimal))
println(@sprintf("  SDDP optimal xO₁ : %d shares  (approx obj %.6f)",
    sddp_opt_xO1, sddp_total[sddp_opt_xO1 + 1]))

using Plots

# ── Plot 1: Stage-1 recourse function (expected future cost, stages 2–4) ─────
plt_recourse = plot(
    n_orders, dp_future;
    label     = "Exact DP",
    lw        = 2,
    color     = :steelblue,
    xlabel    = "Shares purchased at stage 1 (xO₁)",
    ylabel    = "Expected future cost (stages 2–4)",
    title     = "Stage-1 recourse function",
    legend    = :topright,
)
plot!(plt_recourse, n_orders, sddp_future;
    label     = "SDDP lower approx.",
    lw        = 2,
    color     = :darkorange,
    linestyle = :dash,
)
vline!(plt_recourse, [dp_opt_xO1];
    label     = "DP optimum (xO₁=$(dp_opt_xO1))",
    lw        = 1.5,
    color     = :steelblue,
    linestyle = :dot,
)

# ── Plot 2: Total expected cost = stage-1 immediate cost + recourse ───────────
plt_total = plot(
    n_orders, dp_total;
    label     = "Exact DP total cost",
    lw        = 2,
    color     = :steelblue,
    xlabel    = "Shares purchased at stage 1 (xO₁)",
    ylabel    = "Expected total cost",
    title     = "Total expected cost vs first-stage order",
    legend    = :topright,
)
plot!(plt_total, n_orders, sddp_total;
    label     = "SDDP lower approx. total cost",
    lw        = 2,
    color     = :darkorange,
    linestyle = :dash,
)
vline!(plt_total, [dp_opt_xO1];
    label     = "DP optimum (xO₁=$(dp_opt_xO1))",
    lw        = 1.5,
    color     = :steelblue,
    linestyle = :dot,
)
vline!(plt_total, [sddp_opt_xO1];
    label     = "SDDP optimum (xO₁=$(sddp_opt_xO1))",
    lw        = 1.5,
    color     = :darkorange,
    linestyle = :dot,
)

display(plot(plt_recourse, plt_total; layout = (1, 2), size = (950, 420)))
