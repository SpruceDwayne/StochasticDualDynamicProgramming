using LinearAlgebra
using JuMP
using MathOptInterface
const MOI = MathOptInterface


"Forward pass artifacts."
struct ForwardRecord
    x_state::Vector{Vector{Vector{Float64}}}  # x_state[s][t]
    ctx::Vector{Vector{Any}}                  # ctx[s][t]  (node context at stage t)
end

"""
forward_pass!(m::SDDP; trajectories, x0, ctx0=nothing)

Rolls the current policy. We record visited states x_t and their node contexts ctx_t.
"""

function forward_pass!(m::SDDP; trajectories::Vector{Trajectory}, x0::Vector{Float64}, ctx0 = nothing)
    T = m.T
    S = length(trajectories)
    x_hist   = [ [zeros(m.stages[t].state_dim) for t in 1:T] for _ in 1:S ]
    ctx_hist = [ [ctx0 for _ in 1:T] for _ in 1:S ]

    for (s, tr) in enumerate(trajectories)
    x   = copy(x0)
    ctx = ctx0
    for t in 1:T
        stg     = m.stages[t]
        ωt      = tr.ω[t]
        vf_next = m.V[min(t+1, T)]

        # --- record PRE-decision state as the support point for stage t
        x_hist[s][t]   = copy(x)      # <- store x_t (not x_{t+1})
        ctx_hist[s][t] = ctx

        model, x_state, θ, misc = stg.build(t, vf_next, ωt; fix_state = x)  # anchor at x_t
        optimize!(model)

        # advance state
        x = haskey(misc, :x_next) ? value.(misc[:x_next]) : value.(x_state)
        ctx = stg.next_ctx(t, ctx, ωt)
        end
    end
    return ForwardRecord(x_hist, ctx_hist)
end

function forward_pass_online!(m::SDDP; S::Int, x0::Vector{Float64}, ctx0 = nothing)
    T = m.T
    x_hist   = [ [zeros(m.stages[t].state_dim) for t in 1:T] for _ in 1:S ]
    ctx_hist = [ [ctx0 for _ in 1:T] for _ in 1:S ]

    for s in 1:S
    x   = copy(x0)
    ctx = ctx0
    for t in 1:T
        stg     = m.stages[t]
        ωt      = stg.sampler()
        vf_next = m.V[min(t+1, T)]

        # --- record PRE-decision support point
        x_hist[s][t]   = copy(x)
        ctx_hist[s][t] = ctx

        model, x_state, θ, misc = stg.build(t, vf_next, ωt; fix_state = x)  # anchor at x_t
        optimize!(model)

        x   = haskey(misc, :x_next) ? value.(misc[:x_next]) : value.(x_state)
        ctx = stg.next_ctx(t, ctx, ωt)
        end
    end
    return ForwardRecord(x_hist, ctx_hist)
end

# -----------------------
# Dual-based cut from θ-epigraph + dynamics
# -----------------------
# Return the coefficient of `var` in a scalar affine constraint `cref`.
# Works for both MOI.ScalarAffineFunction and (if present) JuMP.GenericAffExpr.
function _coef_in(cref::JuMP.ConstraintRef, var::JuMP.VariableRef)
    # Get the underlying function of the constraint
    f = JuMP.constraint_object(cref).func
    if f isa MOI.ScalarAffineFunction{Float64}
        vidx = JuMP.index(var)  # MOI.VariableIndex to compare against term.variable
        s = 0.0
        @inbounds for term in f.terms
            if term.variable == vidx
                s += term.coefficient
            end
        end
        return s
    elseif f isa JuMP.GenericAffExpr
        return JuMP.coefficient(f, var)
    else
        error("Constraint is not affine in a supported form: $(typeof(f))")
    end
end


"""
compute_cut!(t, model, stg, vf_next, x_support, ω, misc) -> (α::Float64, β::Vector{Float64})

Return a stage-t supporting cut for Vₜ at the support point `x_support`:
    Vₜ(x) ≥ α + β' x

This function supports TWO linear formulations. It picks the first one whose
artifacts are present in `misc`:

1) Linking-constraints route (classic/control-with-linking):
   Use this when the stage links the *current state* xₜ to decisions via a linear block
       T * x_state + W * u  ≥  h   (or equality with slack),
   and V_{t+1} depends on the *next argument* (often the decision or an affine image of it).

   Required in `misc`:
     :T         :: AbstractMatrix        # Jacobian wrt current state xₜ (columns = state_dim)
     :link_eq   :: Vector{ConstraintRef} # OR
     :link_ge   :: Vector{ConstraintRef} #    the linking constraints

   Algorithm:
     λ = dual.(misc[:link_eq]) or dual.(misc[:link_ge])
     β = - T' * λ
     if :c_x present, β += misc[:c_x]    # linear cost term on the current state
     α = objective_value(model) - β' * x_support

2) Dynamics-equality route (state-as-variable):
   Use this when the builder writes (affine) dynamics explicitly as equalities
       x_next = f(x_state, u, ω) = A * x_state + B * u + d
   and exposes those equalities.

   Required in `misc`:
     :x_state  :: Vector{VariableRef}    # the current-state variables
     :xnext_eq :: Vector{ConstraintRef}  # constraints of the form x_next - f(...) == 0

   Algorithm:
     π = dual.(xnext_eq)
     For each state component j:
        β[j] = - Σ_i π[i] * coefficient(x_state[j] in xnext_eq[i])   # i.e., - (∂f/∂x)ᵀ π
     if :c_x present, β += misc[:c_x]
     α = objective_value(model) - β' * x_support

Common requirements/notes:
  • `x_support` must have length `stg.state_dim` and be the point where Vₜ is linearized.
  • The model must be solved to optimality before calling (LP status `MOI.OPTIMAL`).
  • `:c_x` (optional) is the linear cost on the current state in the stage objective
    (∂g/∂xₜ); if absent it’s treated as zero.
  • All computations assume linear/affine constraints and costs.

What to stash in `misc` (examples):

  # Linking route
  misc = Dict(
      :x_state  => x_state,                    # optional, not used by the formula
      :x_next   => x_next,                     # used by forward pass to read next state
      :link_ge  => link_constraints,           # or :link_eq if you used equality+slack
      :T        => T_matrix,                   # multiplies current state in linking block
      :c_x      => zeros(stg.state_dim),       # put nonzeros if stage cost has cₓ' x_state
  )

  # Dynamics route
  misc = Dict(
      :x_state  => x_state,                    # current-state variables
      :x_next   => x_next,                     # used by forward pass
      :xnext_eq => xnext_equalities,           # x_next - f(x_state, u, ω) == 0
      :c_x      => zeros(stg.state_dim),
  )
"""
function compute_cut!(t::Int, model, stg::Stage, vf_next::ValueFn{Float64},
                      x_support::Vector{Float64}, ω, misc)::Tuple{Float64,Vector{Float64}}

    n = stg.state_dim

    # Helper: normalize constraint(s) from misc[key] to a Vector{<:ConstraintRef}
    _get_crefs = function (key::Symbol)
        @assert haskey(misc, key) "compute_cut!: misc must contain $(key)"
        v = misc[key]
        if v isa JuMP.ConstraintRef
            return [v]
        elseif v isa AbstractVector{<:JuMP.ConstraintRef}
            return collect(v)
        else
            error("compute_cut!: misc[:$key] must be a ConstraintRef or a vector of them; got $(typeof(v))")
        end
    end

    # --- 1) Linking-constraints route (if provided)
    if haskey(misc, :T) && (haskey(misc, :link_eq) || haskey(misc, :link_ge))
        linkkey = haskey(misc, :link_eq) ? :link_eq : :link_ge
        link_cref = _get_crefs(linkkey)
        λ = dual.(link_cref)
        T = misc[:T] :: AbstractMatrix
        @assert size(T, 2) == n "T must have n=state_dim columns; got $(size(T)) vs $n"
        β = -transpose(T) * collect(λ)
        β = Vector{Float64}(β)
        if haskey(misc, :c_x) && !isempty(misc[:c_x])
            β .+= misc[:c_x]
        end
        α = objective_value(model) - dot(β, x_support)
        return (α, β)
    end

    # --- 2) Dynamics-equality route (fallback)
    @assert haskey(misc, :x_state)  "builder must provide :x_state (or :T & :link_*)"
    x_state = misc[:x_state] :: Vector{JuMP.VariableRef}
    @assert length(x_state) == n "x_state length $(length(x_state)) ≠ state_dim $n"

    xnext_eq = _get_crefs(:xnext_eq)
    π = dual.(xnext_eq)

    β = zeros(Float64, n)
    # β = - (∂f/∂x)^T * π   recovered from equality coefficients
    for j in 1:n
        acc = 0.0
        for i in 1:length(xnext_eq)
            #a_ij = JuMP.coefficient(xnext_eq[i], x_state[j])  # = -(∂f_i/∂x_j)
            a_ij = _coef_in(xnext_eq[i], x_state[j])           # = -(∂f_i/∂x_j)
            acc += π[i] * a_ij
        end
        β[j] = -acc
    end

    if haskey(misc, :c_x) && !isempty(misc[:c_x])
        β .+= misc[:c_x]
    end

    α = objective_value(model) - dot(β, x_support)
    return (α, β)
end
# -----------------------
# Helper: bucket paths by node id at stage t (so we add ONE expected cut per node)
# -----------------------



# Helper: group scenario indices by node key at stage t
function group_by_node_from_ctx(fwd::ForwardRecord, stages::Vector{Stage}, t::Int)
    buckets = Dict{Any, Vector{Int}}()
    stg = stages[t]
    for s in 1:length(fwd.x_state)
        key = stg.node_key(fwd.ctx[s][t])  # user ensures this clusters equal nodes
        push!(get!(buckets, key, Int[]), s)
    end
    return buckets
end

"""
backward_pass_expected!(m::SDDP; fwd, iter::Int=1, force_every::Int=10, atol::Float64=1e-8)

Expected-cut backward pass.
- Adds a cut at a node if it improves V_t at the node's support point by > atol.
- Additionally, on iterations divisible by `force_every`, it adds the cut
  unconditionally (even if inactive at the support). Set force_every <= 0 to disable.
"""
function backward_pass_expected!(m::SDDP; fwd::ForwardRecord, iter::Int=1,
                                 force_every::Int=10, atol::Float64=1e-8)
    T = m.T
    for t in T:-1:1
        vf_next = m.V[min(t+1, T)]
        stg = m.stages[t]

        buckets = group_by_node_from_ctx(fwd, m.stages, t)

        for (_key, scen_idx) in buckets
            # representative support point x_t at this node
            s₁ = first(scen_idx)
            x_support = fwd.x_state[s₁][t]
            ctx       = fwd.ctx[s₁][t]

            Ωs, ps = stg.children(t, ctx)
            @assert length(Ωs) == length(ps)
            @assert abs(sum(ps) - 1.0) < 1e-12

            α_acc = 0.0
            β_acc = zeros(stg.state_dim)

            for (ω, pω) in zip(Ωs, ps)
                model, x_state, θ, misc = stg.build(t, vf_next, ω; fix_state = x_support)
                optimize!(model)
                misc[:x_state] = get(misc, :x_state, x_state)

                α, β = compute_cut!(t, model, stg, vf_next, x_support, ω, misc)
                α_acc += pω * α
                β_acc .+= pω .* β
            end

            # --- Activity test at support point (with periodic forced adds)
            val_old, _ = evaluate(m.V[t], x_support)
            val_new     = α_acc + dot(β_acc, x_support)
            should_force = force_every > 0 && (iter % force_every == 0)
            if should_force || (val_new > val_old + atol)
                add_cut!(m.V[t], α_acc, β_acc, t)
            end
        end
    end
    return nothing
end


################Markov version below##############
function forward_pass_markov_online!(m::MarkovSDDP;
                                     S::Int,
                                     x0::Vector{Float64},
                                     ctx0)
    T = m.T
    # x_state[s][t] and ctx[s][t], same layout as ForwardRecord requires
    x_hist   = [ [zeros(m.stages[t].state_dim) for t in 1:T] for _ in 1:S ]
    ctx_hist = [ [ctx0 for _ in 1:T] for _ in 1:S ]

    for s in 1:S
        x   = copy(x0)
        ctx = ctx0  # Markov state z_t
        for t in 1:T
            stg = m.stages[t]

            # Sample shock conditional on current Markov state
            ωt = stg.sampler(ctx)  # e.g. WeatherShock

            # Pick a continuation value function for policy evaluation in forward pass.
            vf_next = (t < T) ? get_V!(m, t+1, ctx) : ValueFn{Float64}()

            # Record PRE-decision (x_t, z_t)
            x_hist[s][t]   = copy(x)
            ctx_hist[s][t] = ctx

            model, x_state, θ, misc = stg.build(t, vf_next, ωt; fix_state=x)
            optimize!(model)

            x   = haskey(misc, :x_next) ? value.(misc[:x_next]) : value.(x_state)
            #update future state/ctx such that when we run stg.sampler(ctx) in the next iteration of this loop we will sample from that new state/ctx
            ctx = stg.next_ctx(t, ctx, ωt)
        end
    end
    return ForwardRecord(x_hist, ctx_hist)
end


function backward_pass_markov_expected!(m::MarkovSDDP;
                                        fwd::ForwardRecord,
                                        iter::Int = 1,
                                        force_every::Int = 10,
                                        atol::Float64 = 1e-8)
    T = m.T
    for t in T:-1:1
        stg = m.stages[t]

        # Bucket scenarios by node key at stage t (here: Markov state)
        buckets = group_by_node_from_ctx(fwd, m.stages, t)

        for (_key, scen_idx) in buckets
            # Representative support point at this node
            s₁        = first(scen_idx)
            x_support = fwd.x_state[s₁][t]
            ctx       = fwd.ctx[s₁][t]  # current Markov state z_t

            # Children shocks and probabilities (joint over z_{t+1}, ω)
            Ωs, ps = stg.children(t, ctx)
            @assert length(Ωs) == length(ps)
            @assert abs(sum(ps) - 1.0) < 1e-12

            α_acc = 0.0
            β_acc = zeros(stg.state_dim)

            # For each child shock, route through the correct continuation V_{t+1}^{ctx_next}
            for (ω, pω) in zip(Ωs, ps)
                ctx_next = stg.next_ctx(t, ctx, ω)

                vf_next = if t < T
                    get_V!(m, t+1, ctx_next)  # create V_{t+1}^{ctx_next} lazily if needed
                else
                    ValueFn{Float64}()        # no continuation at final stage
                end

                model, x_state, θ, misc = stg.build(t, vf_next, ω; fix_state=x_support)
                optimize!(model)
                misc[:x_state] = get(misc, :x_state, x_state)

                α, β = compute_cut!(t, model, stg, vf_next, x_support, ω, misc)

                α_acc += pω * α
                β_acc .+= pω .* β
            end

            # Activity test for THIS Markov state's value function V_t^{ctx}
            vf_t_ctx = get_V!(m, t, ctx)
            val_old, _ = evaluate(vf_t_ctx, x_support)
            val_new    = α_acc + dot(β_acc, x_support)

            should_force = force_every > 0 && (iter % force_every == 0)
            if should_force || (val_new > val_old + atol)
                add_cut!(vf_t_ctx, α_acc, β_acc, t)
            end
        end
    end
    return nothing
end

##########AVaR below#############
#Adjust backward_pass_markov_expected such that instead of weighting cuts under the real world measure probabilities pω we solve the dual formulation to obtain new probabilities pw*zetaw
"""
    _avar_dual_weights(ps, vals; alpha)

Given probabilities `ps` (sum to 1) and scenario values `vals` (e.g. V_{t+1} at x_support),
compute the optimal dual weights for AVaR_α using:

    AVaR_α(Z) = max_{ζ}  Σ p_ω ζ_ω Z_ω
                s.t.    Σ p_ω ζ_ω = 1
                        0 ≤ ζ_ω ≤ 1/(1-α)

We return λ_ω = p_ω * ζ_ω so that Σ λ_ω = 1 and

    AVaR_α(Z) = Σ λ_ω Z_ω.

`alpha` ∈ [0,1) is the AVaR level.
"""
function _avar_dual_weights(ps::AbstractVector{<:Real},
                            vals::AbstractVector{<:Real};
                            alpha::Float64)
    @assert length(ps) == length(vals)
    @assert 0.0 ≤ alpha < 1.0
    @assert abs(sum(ps) - 1.0) < 1e-10 "Probabilities must sum to 1."

    n = length(ps)
    λ = zeros(Float64, n)

    # Capacity per scenario: λ_ω ≤ p_ω / (1-α)
    cap_scale = 1.0 / (1.0 - alpha)

    # Greedy "fractional knapsack": allocate probability mass to worst scenarios
    idx = sortperm(vals; rev = true)  # sort by vals descending (worst/biggest)
    remaining = 1.0

    for k in idx
        cap = ps[k] * cap_scale
        assign = min(cap, remaining)
        λ[k] = assign
        remaining -= assign
        if remaining ≤ 1e-12
            break
        end
    end

    @assert remaining ≤ 1e-6 "AVaR dual weights: capacity insufficient; check ps and alpha."
    return λ
end




"""
    backward_pass_markov_rho!(m::MarkovSDDP;
                              fwd::ForwardRecord,
                              iter::Int = 1,
                              force_every::Int = 10,
                              atol::Float64 = 1e-8,
                              alpha::Float64,
                              lambda::Float64)

Markov backward pass under the combined risk measure

    ρ_{α,λ}(Z) = λ E[Z] + (1-λ) AVaR_α(Z).

At each visited node (t, ctx) and support point x_t:

1. For each child shock ω, build a scenario cut (α_ω, β_ω) and compute
   the scenario value at the support point: v_ω = α_ω + β_ω' x_t.

2. Compute AVaR dual weights λ^{AVaR}_ω using `_avar_dual_weights(ps, v_ω; alpha)`.

3. Form combined weights
       w_ω = λ * p_ω + (1-λ) * λ^{AVaR}_ω,
   which satisfy Σ_ω w_ω = 1.

4. Aggregate one cut:
       V_t^{ctx}(x) ≥ Σ_ω w_ω (α_ω + β_ω' x)
                    = ᾱ + β̄' x

   with ᾱ = Σ_ω w_ω α_ω,  β̄ = Σ_ω w_ω β_ω.

Special cases:
- λ = 1 gives the risk-neutral backward pass.
- λ = 0 gives a pure AVaR backward pass.
"""
function backward_pass_markov_rho!(m::MarkovSDDP;
                                   fwd::ForwardRecord,
                                   iter::Int = 1,
                                   force_every::Int = 10,
                                   atol::Float64 = 1e-8,
                                   alpha::Float64,
                                   lambda::Float64)

    @assert 0.0 ≤ lambda ≤ 1.0 "lambda must be in [0,1]."
    @assert 0.0 ≤ alpha  < 1.0 "alpha must be in [0,1)."

    T = m.T
    for t in T:-1:1
        stg = m.stages[t]

        # Bucket scenarios by node key at stage t (Markov state / context)
        buckets = group_by_node_from_ctx(fwd, m.stages, t)

        for (_key, scen_idx) in buckets
            # Representative support point at this node
            s₁        = first(scen_idx)
            x_support = fwd.x_state[s₁][t]
            ctx       = fwd.ctx[s₁][t]  # Markov state z_t

            # Children shocks and probabilities under the real-world measure
            Ωs, ps = stg.children(t, ctx)
            @assert length(Ωs) == length(ps)
            @assert abs(sum(ps) - 1.0) < 1e-12

            nω = length(Ωs)
            α_vec = zeros(Float64, nω)
            β_vec = [zeros(Float64, stg.state_dim) for _ in 1:nω]
            vals  = zeros(Float64, nω)

            # --- Build scenario cuts ---
            for (k, (ω, pω)) in enumerate(zip(Ωs, ps))
                ctx_next = stg.next_ctx(t, ctx, ω)

                vf_next = if t < T
                    get_V!(m, t+1, ctx_next)
                else
                    ValueFn{Float64}()  # no continuation at final stage
                end

                model, x_state, θ, misc = stg.build(t, vf_next, ω; fix_state = x_support)
                optimize!(model)
                misc[:x_state] = get(misc, :x_state, x_state)

                αω, βω = compute_cut!(t, model, stg, vf_next, x_support, ω, misc)
                α_vec[k] = αω
                β_vec[k] = βω
                vals[k]  = αω + dot(βω, x_support)  # scenario value at support point
            end

            # --- AVaR part: λ_avar (probabilities summing to 1) ---
            λ_avar = _avar_dual_weights(ps, vals; alpha = alpha)

            # --- Combined weights for ρ_{α,λ} ---
            w = lambda .* ps .+ (1.0 - lambda) .* λ_avar
            @assert abs(sum(w) - 1.0) < 1e-10

            # Aggregate combined-risk cut
            α_acc = 0.0
            β_acc = zeros(Float64, stg.state_dim)
            for k in 1:nω
                α_acc += w[k] * α_vec[k]
                β_acc .+= w[k] .* β_vec[k]
            end

            # Activity test for THIS Markov state's value function V_t^{ctx}
            vf_t_ctx = get_V!(m, t, ctx)
            val_old, _ = evaluate(vf_t_ctx, x_support)
            val_new    = α_acc + dot(β_acc, x_support)

            should_force = force_every > 0 && (iter % force_every == 0)
            if should_force || (val_new > val_old + atol)
                add_cut!(vf_t_ctx, α_acc, β_acc, t)
            end
        end
    end
    return nothing
end


