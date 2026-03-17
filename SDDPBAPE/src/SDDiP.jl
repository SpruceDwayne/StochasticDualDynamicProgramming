using LinearAlgebra
using JuMP
using Printf

# ============================================================================
# Configuration
# ============================================================================

"""
    SDDiPConfig

Configuration for the SDDiP algorithm.

Fields:
- `cut_type`       – which cuts to generate:
    * `:SB`         Strengthened Benders (one Lagrangian solve with π_LP; valid & finite)
    * `:lagrangian` Full Lagrangian dual via subgradient (valid, tight & finite)
    * `:IO`         Integer Optimality cut (tight only at evaluated point)
    * `:SB_IO`      Both SB and IO cuts per backward step 
- `lag_tol`        – subgradient convergence tolerance (default 1e-4)
- `lag_max_iter`   – max subgradient iterations per child (default 200)
- `step_size_init` – initial subgradient step size (default 1.0)
- `step_decay`     – multiplicative step-size decay per iteration (default 0.95)
"""
struct SDDiPConfig
    cut_type::Symbol
    lag_tol::Float64
    lag_max_iter::Int
    step_size_init::Float64
    step_decay::Float64
end

function SDDiPConfig(;
    cut_type::Symbol      = :lagrangian,#SB,
    lag_tol::Float64      = 1e-4,
    lag_max_iter::Int     = 200,
    step_size_init::Float64 = 1.0,
    step_decay::Float64   = 0.95,
)
    @assert cut_type in (:SB, :lagrangian, :IO, :SB_IO) "cut_type must be one of :SB, :lagrangian, :IO, :SB_IO; got :$cut_type"
    SDDiPConfig(cut_type, lag_tol, lag_max_iter, step_size_init, step_decay)
end

# ============================================================================
# Lagrangian helpers
# ============================================================================

"""
    _get_lp_dual(model, z_vars) -> (π, lp_obj)

Temporarily relax integrality of `model`, solve the resulting LP (with z_vars already
fixed to x_parent by get_or_build_model!), extract the dual of each fixing constraint,
then restore integrality.

Returns:
- `π`      – dual multipliers for z_vars[i] = x_parent[i]   (length d)
- `lp_obj` – LP optimal objective value (a lower bound on the MIP value)
"""
function _get_lp_dual(model::JuMP.Model, z_vars::Vector{JuMP.VariableRef})
    d = length(z_vars)

    # Verify the contract: z_vars must be fixed via JuMP.fix so that FixRef and
    # its dual are well-defined.  This catches stale refs or alternative fixing
    # strategies before they produce a wrong (silent) dual value.
    for i in 1:d
        JuMP.is_fixed(z_vars[i]) || error(
            "SDDiP._get_lp_dual: z_vars[$i] ($(z_vars[i])) is not fixed. " *
            "All z_vars must be fixed via JuMP.fix before calling _get_lp_dual. " *
            "Check that get_or_build_model! (or _solve_lagrangian_subproblem!) " *
            "has re-fixed z_vars to x_parent before this call."
        )
    end

    undo = JuMP.relax_integrality(model)
    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    if status != MOI.OPTIMAL
        undo()
        error("SDDiP: LP relaxation terminated with status $status (expected OPTIMAL)")
    end
    if !JuMP.has_duals(model)
        undo()
        error("SDDiP: LP solver did not return dual values; check solver settings")
    end

    lp_obj = JuMP.objective_value(model)
    π = [JuMP.dual(JuMP.FixRef(z_vars[i])) for i in 1:d]

    undo()
    return π, lp_obj
end


"""
    _solve_lagrangian_subproblem!(model, z_vars, π, x_parent) -> (L_val, z_sol)

Solve the Lagrangian subproblem

    L(π) = min { f(x,y) + θ - π'z : constraints, x ∈ {0,1}^d, z ∈ [0,1]^d }

by:
1. Unfixing z_vars and restoring [0,1] bounds.
2. Adding –π[i] as the objective coefficient for z_vars[i].
3. Solving the resulting MIP.
4. Restoring: reset z objective coefficients to 0 and re-fix z_vars to x_parent.

Returns the Lagrangian value L(π) and the optimal z solution.
"""
function _solve_lagrangian_subproblem!(
    model::JuMP.Model,
    z_vars::Vector{JuMP.VariableRef},
    π::Vector{Float64},
    x_parent::Vector{Float64},
)
    d = length(z_vars)

    # Unfix z_vars and restore bounds
    for i in 1:d
        JuMP.unfix(z_vars[i])
        JuMP.set_lower_bound(z_vars[i], 0.0)
        JuMP.set_upper_bound(z_vars[i], 1.0)
    end

    # Add –π'z to the objective
    for i in 1:d
        JuMP.set_objective_coefficient(model, z_vars[i], -π[i])
    end

    JuMP.optimize!(model)
    L_val  = JuMP.objective_value(model)
    z_sol  = JuMP.value.(z_vars)

    # Restore: zero out z objective coefficients and re-fix to x_parent
    for i in 1:d
        JuMP.set_objective_coefficient(model, z_vars[i], 0.0)
        JuMP.fix(z_vars[i], x_parent[i]; force = true)
    end

    return L_val, z_sol
end


"""
    solve_lagrangian_dual!(model, z_vars, x_parent, π_init, config) -> (best_L, best_π)

Maximise the Lagrangian dual function

    g(π) = L(π) + π'x_parent

via subgradient ascent, starting from `π_init` (typically the LP dual).

Subgradient of g at π: ∇g = x_parent − z*(π).
Convergence: ‖∇g‖ < config.lag_tol or config.lag_max_iter reached.

Returns the best dual bound `best_L` and associated multiplier `best_π`,
such that the cut  θ ≥ best_L + best_π'x  is valid for all x ∈ {0,1}^d.
"""
function solve_lagrangian_dual!(
    model::JuMP.Model,
    z_vars::Vector{JuMP.VariableRef},
    x_parent::Vector{Float64},
    π_init::Vector{Float64},
    config::SDDiPConfig,
)
    π      = copy(π_init)
    best_g = -Inf   # tracks g(π) = L(π) + π'x_parent to find the best π
    best_L = -Inf   # L(π*) — the actual cut intercept α (NOT g(π*))
    best_π = copy(π)
    step   = config.step_size_init

    for _ in 1:config.lag_max_iter
        L_val, z_sol = _solve_lagrangian_subproblem!(model, z_vars, π, x_parent)

        obj = L_val + dot(π, x_parent)   # g(π) — Lagrangian dual bound
        if obj > best_g
            best_g = obj
            best_L = L_val   # store L(π*), not g(π*), so the cut θ ≥ best_L + best_π'x is valid
            best_π = copy(π)
        end

        subgrad = x_parent .- z_sol
        norm(subgrad) < config.lag_tol && break

        π    .+= step .* subgrad
        step  *= config.step_decay
    end

    return best_L, best_π
end

# ============================================================================
# Integer Optimality (IO) cut coefficients
# ============================================================================

"""
    _io_cut_coefficients(v_bar, lp_obj, x_bar) -> (α, β)

Compute the Integer Optimality cut (affine in x) that is tight at x̄ ∈ {0,1}^d:

    θ ≥ v̄ − (v̄ − L̲) [ Σ_{j:x̄ⱼ=1}(1−xⱼ)  +  Σ_{j:x̄ⱼ=0} xⱼ ]

where L̲ = lp_obj (LP lower bound) and v̄ = v_bar (MIP optimal value at x̄).

In affine form θ ≥ α + β'x:
    α   = v̄ − (v̄−L̲)·‖x̄‖₁
    β[j] = (v̄−L̲)·(2x̄[j] − 1)

Valid for all x ∈ {0,1}^d; tight only at x = x̄.
"""
function _io_cut_coefficients(
    v_bar::Float64,
    lp_obj::Float64,
    x_bar::AbstractVector{Float64},
)
    gap = max(v_bar - lp_obj, 0.0)   # MIP ≥ LP for minimisation → gap ≥ 0
    α   = v_bar - gap * sum(x_bar)
    β   = [gap * (2.0 * x_bar[j] - 1.0) for j in eachindex(x_bar)]
    return α, β
end

# ============================================================================
# Main SDDiP cut computation
# ============================================================================

"""
    compute_sddip_cut!(model, z_vars, x_support, config) -> cuts

Compute one or two SDDiP cuts for the child subproblem whose z_vars are already
fixed to `x_support` (done by get_or_build_model!).

Returns a `Vector{Tuple{Float64,Vector{Float64}}}` — one (α,β) pair per cut generated:
- `:SB`         → 1 cut  (Strengthened Benders)
- `:lagrangian` → 1 cut  (full Lagrangian dual)
- `:IO`         → 1 cut  (Integer Optimality)
- `:SB_IO`      → 2 cuts (SB + IO)

The returned cuts all satisfy  Vₜ₊₁(x) ≥ α + β'x  for x ∈ {0,1}^d.
"""
function compute_sddip_cut!(
    model::JuMP.Model,
    z_vars::Vector{JuMP.VariableRef},
    x_support::AbstractVector{Float64},
    config::SDDiPConfig,
)
    x_par  = collect(Float64, x_support)
    result = Tuple{Float64, Vector{Float64}}[]

    # Step 1: LP relaxation → dual multiplier π_LP and LP lower bound
    π, lp_obj = _get_lp_dual(model, z_vars)

    if config.cut_type === :lagrangian
        # Full Lagrangian dual via subgradient ascent
        best_L, best_π = solve_lagrangian_dual!(model, z_vars, x_par, π, config)
        push!(result, (best_L, best_π))

    elseif config.cut_type === :SB || config.cut_type === :SB_IO
        # Strengthened Benders: one Lagrangian solve with π = π_LP
        L_val, _ = _solve_lagrangian_subproblem!(model, z_vars, π, x_par)
        push!(result, (L_val, copy(π)))

        if config.cut_type === :SB_IO
            # Integer Optimality: full MIP solve with z fixed to x_parent
            # z_vars were re-fixed by _solve_lagrangian_subproblem!
            JuMP.optimize!(model)
            v_bar        = JuMP.objective_value(model)
            α_io, β_io   = _io_cut_coefficients(v_bar, lp_obj, x_par)
            push!(result, (α_io, β_io))
        end

    elseif config.cut_type === :IO
        # Integer Optimality only: z_vars already fixed → solve MIP directly
        JuMP.optimize!(model)
        v_bar      = JuMP.objective_value(model)
        α_io, β_io = _io_cut_coefficients(v_bar, lp_obj, x_par)
        push!(result, (α_io, β_io))
    end

    #Make this new option such that we can compare LP cuts 
    #elseif config.cut_type ===B

    return result
end

# ============================================================================
# SDDiP backward passes
# ============================================================================

"""
    backward_pass_sddip!(m::SDDP; fwd, config, iter, force_every, atol)

SDDiP backward pass for the IID (non-Markov) case.

For each visited node at stage t:
1. Retrieve all child scenarios (Ωs, ps) from the stage.
2. For each child ω, build/fetch the cached subproblem (z_vars fixed to x_support).
3. Call compute_sddip_cut! to generate (α,β) pairs.
4. Accumulate expected cuts: α_acc += pω·α, β_acc += pω·β.
5. Add each accumulated cut to V[t] if it improves the current envelope.

Requirement: every stage builder must expose `misc[:z_vars]` (the continuous copy
variables z ∈ [0,1]^d of the parent binary state).
"""
function backward_pass_sddip!(
    m::SDDP;
    fwd::ForwardRecord,
    config::SDDiPConfig,
    iter::Int        = 1,
    force_every::Int = 10,
    atol::Float64    = 1e-8,
)
    T          = m.T
    n_cut_types = config.cut_type === :SB_IO ? 2 : 1

    for t in T:-1:1
        # At the terminal stage there is no continuation; use an empty VF so that
        # newly-added SDDiP cuts are not fed back as epigraph constraints into the
        # terminal model (which has θ == 0 and would become infeasible otherwise).
        vf_next = t < T ? m.V[t + 1] : ValueFn{Float64}()
        stg     = m.stages[t]

        buckets = group_by_node_from_ctx(fwd, m.stages, t)

        for (_, scen_idx) in buckets
            s₁        = first(scen_idx)
            x_support = fwd.x_state[s₁][t]
            ctx       = fwd.ctx[s₁][t]

            Ωs, ps = stg.children(t, ctx)
            @assert length(Ωs) == length(ps)
            @assert abs(sum(ps) - 1.0) < 1e-12

            α_accs = zeros(Float64, n_cut_types)
            β_accs = [zeros(Float64, stg.state_dim) for _ in 1:n_cut_types]

            for (ω, pω) in zip(Ωs, ps)
                model, _, _, misc = get_or_build_model!(m, t, vf_next, ω, ctx, x_support)
                # Skip stages that have no copy variable (e.g. the initial stage)
                haskey(misc, :z_vars) || continue
                z_vars = misc[:z_vars]

                pairs = compute_sddip_cut!(model, z_vars, x_support, config)
                for (k, (α, β)) in enumerate(pairs)
                    α_accs[k]   += pω * α
                    β_accs[k]  .+= pω .* β
                end
            end

            # Add each accumulated cut if it improves the current envelope
            for k in 1:n_cut_types
                val_old, _   = evaluate(m.V[t], x_support)
                val_new       = α_accs[k] + dot(β_accs[k], x_support)
                should_force  = force_every > 0 && (iter % force_every == 0)
                if should_force || (val_new > val_old + atol)
                    add_cut!(m.V[t], α_accs[k], β_accs[k], t)
                end
            end
        end
    end
    return nothing
end


"""
    backward_pass_markov_sddip!(m::MarkovSDDP; fwd, config, iter, force_every, atol)

SDDiP backward pass for the Markov case. Routes continuation value functions through
the correct next Markov state for each child scenario (ω → ctx_next → V_{t+1}^{ctx_next}).

Otherwise identical in structure to backward_pass_sddip!.
"""
function backward_pass_markov_sddip!(
    m::MarkovSDDP;
    fwd::ForwardRecord,
    config::SDDiPConfig,
    iter::Int        = 1,
    force_every::Int = 10,
    atol::Float64    = 1e-8,
)
    T           = m.T
    n_cut_types = config.cut_type === :SB_IO ? 2 : 1

    for t in T:-1:1
        stg     = m.stages[t]
        buckets = group_by_node_from_ctx(fwd, m.stages, t)

        for (_, scen_idx) in buckets
            s₁        = first(scen_idx)
            x_support = fwd.x_state[s₁][t]
            ctx       = fwd.ctx[s₁][t]

            Ωs, ps = stg.children(t, ctx)
            @assert length(Ωs) == length(ps)
            @assert abs(sum(ps) - 1.0) < 1e-12

            α_accs = zeros(Float64, n_cut_types)
            β_accs = [zeros(Float64, stg.state_dim) for _ in 1:n_cut_types]

            for (ω, pω) in zip(Ωs, ps)
                ctx_next = stg.next_ctx(t, ctx, ω)
                vf_next  = t < T ? get_V!(m, t + 1, ctx_next) : ValueFn{Float64}()

                model, _, _, misc = get_or_build_model!(m, t, vf_next, ω, ctx, x_support)
                # Skip stages that have no copy variable (e.g. the initial stage)
                haskey(misc, :z_vars) || continue
                z_vars = misc[:z_vars]

                pairs = compute_sddip_cut!(model, z_vars, x_support, config)
                for (k, (α, β)) in enumerate(pairs)
                    α_accs[k]  += pω * α
                    β_accs[k] .+= pω .* β
                end
            end

            vf_t_ctx = get_V!(m, t, ctx)
            for k in 1:n_cut_types
                val_old, _  = evaluate(vf_t_ctx, x_support)
                val_new      = α_accs[k] + dot(β_accs[k], x_support)
                should_force = force_every > 0 && (iter % force_every == 0)
                if should_force || (val_new > val_old + atol)
                    add_cut!(vf_t_ctx, α_accs[k], β_accs[k], t)
                end
            end
        end
    end
    return nothing
end

# ============================================================================
# SDDiP run loops
# ============================================================================

"""
    run_sddip!(m::SDDP; x0, ctx0, config, S, max_iter, patience,
               value_tol, evaluate_index, rng, force_every, cut_atol, logfn)

SDDiP training loop for the IID (non-Markov) case. Structurally mirrors run_sddp!
but drives backward_pass_sddip! instead of backward_pass_expected!.

All stage builders must expose `misc[:z_vars]` (continuous copy variables).
Recommended: 1–3 forward samples per iteration (S=1 is often best).
"""
function run_sddip!(
    m::SDDP;
    x0::AbstractVector,
    ctx0              = nothing,
    config::SDDiPConfig = SDDiPConfig(),
    S::Int            = 1,
    max_iter::Int     = 1_000,
    patience::Int     = 20,
    value_tol::Float64 = 0.0,
    evaluate_index::Int = 1,
    rng               = Random.default_rng(),
    force_every::Int  = 10,
    cut_atol::Float64 = 1e-8,
    logfn             = nothing,
)
    total_cuts()   = sum(length(m.V[t].cuts) for t in 1:m.T)
    cuts_by_stage()= [length(m.V[t].cuts) for t in 1:m.T]

    prev_total = total_cuts()
    stagnant   = 0
    x0f        = collect(Float64, x0)
    prev_V, _  = evaluate(m.V[evaluate_index], x0f)
    hist       = Vector{NamedTuple}()

    Random.seed!(rng, rand(UInt))

    for it in 1:max_iter
        fwd = forward_pass_online!(m; S = S, x0 = x0f, ctx0 = ctx0)
        backward_pass_sddip!(m; fwd = fwd, config = config, iter = it,
                             force_every = force_every, atol = cut_atol)

        cur_total = total_cuts()
        new_cuts  = cur_total - prev_total
        prev_total = cur_total

        cur_V, _ = evaluate(m.V[evaluate_index], x0f)
        ΔV        = abs(cur_V - prev_V)
        prev_V    = cur_V

        stats = (iter = it, new_cuts = new_cuts, total_cuts = cur_total,
                 V = cur_V, ΔV = ΔV, per_stage = cuts_by_stage())
        push!(hist, stats)

        if logfn === nothing
            @printf "iter %4d | new cuts: %2d | total: %3d | V%d(x0)=%.6f | ΔV=%.3e\n" it new_cuts cur_total evaluate_index cur_V ΔV
        else
            logfn(it, stats)
        end

        stagnant = (new_cuts == 0) ? (stagnant + 1) : 0
        if stagnant >= patience
            println("Early stop: no new cuts for $patience consecutive iterations.")
            return (iters = it, cuts_per_stage = cuts_by_stage(), history = hist)
        end
        if value_tol > 0 && ΔV ≤ value_tol
            println("Early stop: |ΔV| ≤ $value_tol.")
            return (iters = it, cuts_per_stage = cuts_by_stage(), history = hist)
        end
    end

    println("Reached max_iter without early stop.")
    return (iters = max_iter, cuts_per_stage = cuts_by_stage(), history = hist)
end


"""
    run_markov_sddip!(m::MarkovSDDP; x0, ctx0, config, S, max_iter, patience,
                      value_tol, evaluate_stage, evaluate_ctx, rng,
                      force_every, cut_atol, logfn)

SDDiP training loop for the Markov case. Mirrors run_markov_sddp! but drives
backward_pass_markov_sddip! instead of backward_pass_markov_expected!.
"""
function run_markov_sddip!(
    m::MarkovSDDP;
    x0::AbstractVector,
    ctx0,
    config::SDDiPConfig  = SDDiPConfig(),
    S::Int               = 1,
    max_iter::Int        = 1_000,
    patience::Int        = 20,
    value_tol::Float64   = 0.0,
    evaluate_stage::Int  = 1,
    evaluate_ctx,
    rng                  = Random.default_rng(),
    force_every::Int     = 10,
    cut_atol::Float64    = 1e-8,
    logfn                = nothing,
)
    total_cuts() = sum(
        t -> sum(vf -> length(vf.cuts), values(m.V[t]); init = 0),
        1:m.T; init = 0,
    )
    cuts_by_stage() = [
        sum(vf -> length(vf.cuts), values(m.V[t]); init = 0)
        for t in 1:m.T
    ]

    prev_total = total_cuts()
    stagnant   = 0
    x0f        = collect(Float64, x0)
    vf_eval    = get_V!(m, evaluate_stage, evaluate_ctx)
    prev_V, _  = evaluate(vf_eval, x0f)
    hist       = Vector{NamedTuple}()

    Random.seed!(rng, rand(UInt))

    for it in 1:max_iter
        fwd = forward_pass_markov_online_old!(m; S = S, x0 = x0f, ctx0 = ctx0)
        backward_pass_markov_sddip!(m; fwd = fwd, config = config, iter = it,
                                    force_every = force_every, atol = cut_atol)

        cur_total = total_cuts()
        new_cuts  = cur_total - prev_total
        prev_total = cur_total

        vf_eval = get_V!(m, evaluate_stage, evaluate_ctx)
        cur_V, _ = evaluate(vf_eval, x0f)
        ΔV        = abs(cur_V - prev_V)
        prev_V    = cur_V

        stats = (iter = it, new_cuts = new_cuts, total_cuts = cur_total,
                 V = cur_V, ΔV = ΔV, per_stage = cuts_by_stage())
        push!(hist, stats)

        if logfn === nothing
            @printf "iter %4d | new cuts: %2d | total: %3d | V%d[%s](x0)=%.6f | ΔV=%.3e\n" it new_cuts cur_total evaluate_stage string(evaluate_ctx) cur_V ΔV
        else
            logfn(it, stats)
        end

        stagnant = (new_cuts == 0) ? (stagnant + 1) : 0
        if stagnant >= patience
            println("Early stop: no new cuts for $patience consecutive iterations.")
            return (iters = it, cuts_per_stage = cuts_by_stage(), history = hist)
        end
        if value_tol > 0 && ΔV ≤ value_tol
            println("Early stop: |ΔV| ≤ $value_tol.")
            return (iters = it, cuts_per_stage = cuts_by_stage(), history = hist)
        end
    end

    println("Reached max_iter without early stop.")
    return (iters = max_iter, cuts_per_stage = cuts_by_stage(), history = hist)
end

# ============================================================================
# Binarization utility
# ============================================================================

"""
    binarize(U, ε = 1.0) -> (coeffs, n_bits)

Return the binary expansion coefficients for approximating x ∈ [0, U] with
precision ε using n_bits binary variables λ ∈ {0,1}^n_bits:

    x ≈ sum(coeffs[i] * λ[i] for i in 1:n_bits)

- **Integer x ∈ {0,…,U}**: use ε = 1.0 for an exact representation.
- **Continuous x ∈ [0,U]**: use small ε for an ε-accurate approximation;
  n_bits = floor(log2(U/ε)) + 1.

Example:
```julia
coeffs, n = binarize(7.0)       # exact binary for x ∈ {0,...,7}: [1,2,4], n=3
coeffs, n = binarize(1.0, 0.1)  # 4-bit approximation of [0,1] with precision 0.1
```
"""
function binarize(U::Real, ε::Real = 1.0)
    @assert U > 0   "U must be positive"
    @assert 0 < ε ≤ U "ε must be in (0, U]"
    n_bits = floor(Int, log2(U / ε)) + 1
    coeffs = [ε * 2.0^(i - 1) for i in 1:n_bits]
    return coeffs, n_bits
end
