using Pkg
Pkg.activate(joinpath(@__DIR__, "SDDPBAPE"))
using SDDPBAPE
using JuMP, HiGHS
using LinearAlgebra, Printf

# ── Parameters (same as smoke test) ───────────────────────────────────────────
const Gmax = 10.0
const d1   = 6.0
const SU   = 2.0
const F    = 0.5
const c    = 1.0
const M    = 100.0

const scenarios = [4.0, 9.0]
const probs     = [0.5, 0.5]

function build_stage1(t, vf_next, ω; fix_state)
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, x1, Bin)
    @variable(model, u1, Bin)
    @variable(model, g1  >= 0)
    @variable(model, shed1 >= 0)
    @variable(model, θ >= 0)
    @constraint(model, g1 + shed1 == d1)
    @constraint(model, g1 <= Gmax * x1)
    @constraint(model, u1 >= x1)
    for cut in vf_next.cuts
        @constraint(model, θ >= cut.α + cut.β[1] * x1)
    end
    @objective(model, Min, SU*u1 + F*x1 + c*g1 + M*shed1 + θ)
    misc = Dict{Symbol,Any}(:x_next => [x1], :x_state => [x1])
    return model, [x1], θ, misc
end

function build_stage2(t, vf_next, ω; fix_state)
    d2 = ω
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, x2, Bin)
    @variable(model, u2, Bin)
    @variable(model, g2  >= 0)
    @variable(model, shed2 >= 0)
    @variable(model, 0 <= z2 <= 1)
    @constraint(model, z2 == fix_state[1])
    @constraint(model, g2 + shed2 == d2)
    @constraint(model, g2 <= Gmax * x2)
    @constraint(model, u2 >= x2 - z2)
    @variable(model, θ == 0)
    @objective(model, Min, SU*u2 + F*x2 + c*g2 + M*shed2 + θ)
    misc = Dict{Symbol,Any}(:z_vars => [z2], :x_next => [x2], :x_state => [x2])
    return model, [x2], θ, misc
end

stage1 = Stage(1, 1, build_stage1, () -> nothing, _ -> 1.0,
               (t, ctx) -> ([nothing], [1.0]),
               (t, ctx, ω) -> nothing, x -> x)
stage2 = Stage(2, 1, build_stage2, () -> scenarios[rand(1:2)], _ -> 1.0,
               (t, ctx) -> (scenarios, probs),
               (t, ctx, ω) -> nothing, x -> x)

println("="^65)
println("SDDiP Lagrangian Cut Verification Test")
println("="^65)

# ── Fresh model ────────────────────────────────────────────────────────────────
m = SDDP([stage1, stage2])
x0 = [0.0]

println("\n--- Forward pass ---")
fwd = forward_pass_online!(m; S=1, x0=x0, ctx0=nothing)
x_support = fwd.x_state[1][1]
x_par = Float64[x_support[1]]
println("  x_parent = $x_par  (stage-1 binary decision)")
@assert x_par[1] ≈ 0.0 || x_par[1] ≈ 1.0  "x_parent not binary"

# ── Build stage-2 model (scenario d2=4.0) ─────────────────────────────────────
println("\n--- Build stage-2 subproblem (scenario d2 = 4.0) ---")
vf_empty = ValueFn{Float64}()
model2, _, _, misc2 = SDDPBAPE.get_or_build_model!(m, 2, vf_empty, 4.0, nothing, x_support)
z_vars = misc2[:z_vars]
@assert JuMP.is_fixed(z_vars[1])
@printf "  z2 fixed to %.1f  (= x_parent)\n" JuMP.fix_value(z_vars[1])

# ── True MIP value at x_parent ────────────────────────────────────────────────
println("\n--- True MIP value at x_parent (z fixed) ---")
JuMP.optimize!(model2)
v_mip = JuMP.objective_value(model2)
@printf "  v_MIP(x_parent=%.0f) = %.6f\n" x_par[1] v_mip

# ── LP dual ───────────────────────────────────────────────────────────────────
println("\n--- LP dual extraction ---")
π_lp, lp_obj = SDDPBAPE._get_lp_dual(model2, z_vars)
@printf "  π_LP = %.6f,  LP_obj = %.6f\n" π_lp[1] lp_obj
println("  PASS: LP dual extracted")

# ── Manual Lagrangian iteration trace ─────────────────────────────────────────
println("\n--- Manual Lagrangian dual trace (first 5 iterations) ---")
println("  Verifying: subgradient = x_parent - z_sol, π updates between iters")

let π_cur = copy(π_lp), step_sz = 1.0, lag_tol = 1e-4
    for iter in 1:min(5, 200)
        L_val, z_sol = SDDPBAPE._solve_lagrangian_subproblem!(model2, z_vars, π_cur, x_par)

        subgrad = x_par .- z_sol
        g_val   = L_val + dot(π_cur, x_par)

        @printf "  iter %d: L=%.4f  π=%.4f  z=%.4f  sg=%.4f  g(π)=%.4f\n" iter L_val π_cur[1] z_sol[1] subgrad[1] g_val

        @assert JuMP.is_fixed(z_vars[1]) "z_vars not re-fixed after iteration $iter"

        if norm(subgrad) < lag_tol
            println("  Converged at iter $iter (subgradient norm < lag_tol)")
            break
        end

        π_old    = copy(π_cur)
        π_cur  .+= step_sz .* subgrad
        step_sz *= 0.95

        if iter == 1
            @assert !isapprox(π_old, π_cur; atol=1e-10) "π did not update after iter 1!"
            @printf "  PASS: π updated (%.4f → %.4f)\n" π_old[1] π_cur[1]
        end
    end
end
println("  PASS: manual trace complete, subgradient = x_parent − z at each step")

# ── Full solve_lagrangian_dual! ────────────────────────────────────────────────
println("\n--- Full solve_lagrangian_dual! (config: lag_tol=1e-4, max_iter=200) ---")
config_lag = SDDiPConfig(cut_type = :lagrangian, lag_tol = 1e-4, lag_max_iter = 200,
                         step_size_init = 1.0, step_decay = 0.95)
π_lp2, _  = SDDPBAPE._get_lp_dual(model2, z_vars)
best_L, best_π = solve_lagrangian_dual!(model2, z_vars, x_par, π_lp2, config_lag)

@printf "  best_L (returned) = %.6f\n" best_L
@printf "  best_π            = %.6f\n" best_π[1]
@printf "  g(best_π) + ...   = best_L = %.6f  (should ≈ v_MIP = %.6f)\n" best_L v_mip

# ── TIGHTNESS CHECK: L(π)+π'x_parent should equal v_MIP at convergence ────────
println("\n--- Check: L(π*) + π*·x_parent ≈ v_MIP (dual bound tightness) ---")
# In the current implementation best_L stores g(π*) = L(π*) + π*'x_parent.
# So this check is: best_L ≈ v_mip.
gap_dual = abs(best_L - v_mip)
@printf "  |best_L − v_MIP| = %.2e\n" gap_dual
if gap_dual < 1e-4
    println("  PASS: Lagrangian dual bound is tight (gap < 1e-4)")
else
    println("  NOTE: Lagrangian dual gap = $gap_dual (may be > 0 for non-tight problems)")
end

# ── CUT VALIDITY CHECK ────────────────────────────────────────────────────────
println("\n--- Check: cut validity  θ ≥ α + β·x_parent  ≤  v_MIP ---")
pairs = compute_sddip_cut!(model2, z_vars, x_support, config_lag)
@assert length(pairs) == 1 "Expected 1 cut pair for :lagrangian, got $(length(pairs))"
cut_α, cut_β = pairs[1]
cut_val_at_xp = cut_α + dot(cut_β, x_par)

@printf "  Cut: θ ≥ %.6f + %.6f·x\n" cut_α cut_β[1]
@printf "  Cut evaluated at x_parent = %.6f\n" cut_val_at_xp
@printf "  v_MIP at x_parent        = %.6f\n" v_mip
@printf "  cut(x_parent) − v_MIP    = %.6e\n" (cut_val_at_xp - v_mip)

# A valid cut must satisfy: cut(x) ≤ V(x) for all x ∈ {0,1}^d
# At x = x_parent this is: cut_α + cut_β'x_parent ≤ v_mip
if cut_val_at_xp ≤ v_mip + 1e-6
    println("  PASS: cut(x_parent) ≤ v_MIP — cut is valid at x_parent")
else
    println("  FAIL: cut(x_parent) > v_MIP — cut is INVALID (overestimates V)")
    println("        Intercept α = g(π*) instead of L(π*); off by best_π'x_parent = $(dot(best_π, x_par))")
    println("        Bug: solve_lagrangian_dual! stores g(π*) in best_L, should store L(π*)")
end

# ── CUT TIGHTNESS CHECK at x_parent ──────────────────────────────────────────
println("\n--- Check: cut tightness at x_parent (cut should be tight when dual is tight) ---")
# The correct Lagrangian cut: θ ≥ L(π*) + π*'x
# Tight at x_parent when L(π*) + π*'x_parent = V(x_parent) (zero dual gap)
# Correct α = best_L - dot(best_π, x_par)  [L(π*) = g(π*) - π*'x_parent]
correct_α = best_L - dot(best_π, x_par)
correct_cut_at_xp = correct_α + dot(best_π, x_par)   # = best_L
@printf "  Correct α = L(π*) = best_L − best_π·x_parent = %.6f\n" correct_α
@printf "  Correct cut(x_parent) = L(π*) + π*·x_parent  = %.6f  (should ≈ v_MIP)\n" correct_cut_at_xp

# ── Explicit x_parent=1 probe — catches Bug 1 even when forward pass gives 0 ──
# The cut θ ≥ α + β'x must satisfy α + β'x ≤ V(x) for ALL x ∈ {0,1}^d.
# When x_parent=1, Bug 1 (storing g(π*) instead of L(π*)) makes cut(x=1) > V(1).
println("\n--- Cut validity at x_parent=1 (explicit probe, d2=4.0) ---")
let m_p = SDDP([stage1, stage2]), xp1 = [1.0]
    model_p, _, _, misc_p = SDDPBAPE.get_or_build_model!(m_p, 2, ValueFn{Float64}(), 4.0, nothing, xp1)
    zv = misc_p[:z_vars]
    @assert JuMP.fix_value(zv[1]) ≈ 1.0 "z2 not fixed to 1"

    # True MIP value at x_parent=1 (z2 already fixed)
    JuMP.optimize!(model_p)
    v_mip_1 = JuMP.objective_value(model_p)
    @printf "  v_MIP(x=1, d2=4) = %.6f\n" v_mip_1

    cfg_p = SDDiPConfig(cut_type = :lagrangian, lag_tol = 1e-4, lag_max_iter = 200,
                        step_size_init = 1.0, step_decay = 0.95)
    pairs_p = compute_sddip_cut!(model_p, zv, xp1, cfg_p)
    @assert length(pairs_p) == 1
    α_p, β_p = pairs_p[1]

    cut_at_1 = α_p + dot(β_p, [1.0])
    cut_at_0 = α_p + dot(β_p, [0.0])
    @printf "  Cut: θ ≥ %.6f + %.6f·x\n" α_p β_p[1]
    @printf "  cut(x=1) = %.6f   v_MIP(1) = %.6f   diff = %+.2e\n" cut_at_1 v_mip_1 (cut_at_1 - v_mip_1)
    @printf "  cut(x=0) = %.6f   v_MIP(0) = 11.5   diff = %+.2e\n" cut_at_0 (cut_at_0 - 11.5)

    ok1 = cut_at_1 ≤ v_mip_1 + 1e-6
    ok0 = cut_at_0 ≤ 11.5    + 1e-6
    if ok1 && ok0
        println("  PASS: Lagrangian cut valid at both x=0 and x=1")
    else
        !ok1 && println("  FAIL: cut(x=1) > v_MIP(1)  — cut overestimates V at x=1 (Bug 1)")
        !ok0 && println("  FAIL: cut(x=0) > v_MIP(0)  — cut overestimates V at x=0")
        error("Cut validity check failed — see above")
    end
end

# ── run_sddip! with Lagrangian cuts ──────────────────────────────────────────
println("\n--- run_sddip! for 5 iterations with :lagrangian cuts ---")
m2 = SDDP([stage1, stage2])  # fresh model to avoid contamination
result = run_sddip!(m2; x0=x0, config=config_lag, max_iter=5, patience=5,
                    force_every=1, evaluate_index=2)
@printf "  Completed %d iterations, cuts in V[2]: %d\n" result.iters result.cuts_per_stage[2]
if result.cuts_per_stage[2] > 0
    println("  PASS: Lagrangian cuts added to V[2]")
else
    println("  FAIL: No Lagrangian cuts added to V[2]")
end

println("\n" * "="^65)
println("Lagrangian verification complete — see FAIL/PASS above.")
println("="^65)
