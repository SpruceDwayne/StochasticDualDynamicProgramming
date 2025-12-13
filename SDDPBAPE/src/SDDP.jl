using LinearAlgebra
using SparseArrays  # if you use SparseMatrixCSC
using Printf   # <-- simplest
using Random



"One affine cut α + β'x tagged with its stage."
struct Cut{T}
    α::T
    β::Vector{T}
    stage::Int
end

"Piecewise-affine value function V_t(x) = max_k α_k + β_k'x."
mutable struct ValueFn{T}
    cuts::Vector{Cut{T}}
end
ValueFn{T}() where {T} = ValueFn{T}(Cut{T}[])

"Evaluate V(x) and pick an active subgradient (β of any maximizer)."
function evaluate(v::ValueFn, x::AbstractVector)
    if isempty(v.cuts)
        return (zero(eltype(x)), zeros(eltype(x), length(x)))
    end
    vals = map(c -> c.α + dot(c.β, x), v.cuts)
    k = argmax(vals)
    return (vals[k], v.cuts[k].β)
end

"Add a cut if it is new/useful (very basic dominance filtering)."
function add_cut!(v::ValueFn{T}, α::T, β::Vector{T}, stage::Int; atol=1e-8) where {T}
    for c in v.cuts
        if abs(c.α - α) ≤ atol && norm(c.β - β) ≤ atol
            return false
        end
    end
    push!(v.cuts, Cut{T}(α, β, stage))
    return true
end

"""
Stage definition: user supplies builder + stochastic interface.

Fields:
- t, state_dim
- build(t, vf_next, ω; fix_state)
- sampler() -> ω
- weight(ω) -> probability/weight
- children(t, ctx) -> (Ωs, ps)  with sum(ps)==1
- next_ctx(t, ctx, ω) -> ctx_{t+1}
- node_key(ctx) -> hashable key for node grouping
"""
struct Stage
    t::Int
    state_dim::Int
    build::Function
    sampler::Function
    weight::Function
    children::Function
    next_ctx::Function
    node_key::Function
end

# Convenience constructors for common cases -------------------------------------

# Case A: Xi_t is constant (no context), ps constant:
function Stage_constant_Xi(; t, state_dim, build, sampler, weight, Xi, pXi)
    _children = (tt, ctx)->begin
        @assert tt == t
        @assert abs(sum(pXi) - 1.0) < 1e-12
        return (Xi, pXi)
    end
    Stage(t, state_dim, build, sampler, weight, _children,
          (tt, ctx, ω)->nothing, # next_ctx
          x->x)                   # node_key (identity; ctx=nothing, so all visits group)
end

# Case B: Xi_t depends on a small discrete Markov state ctx::Int
function Stage_markov_Xi(; t, state_dim, build, sampler, weight, Xi_of, pXi_of, next_ctx, node_key = x->x)
    _children = (tt, ctx)->begin
        @assert tt == t
        Xi = Xi_of(ctx)
        ps = pXi_of(ctx)
        @assert abs(sum(ps) - 1.0) < 1e-12
        return (Xi, ps)
    end
    Stage(t, state_dim, build, sampler, weight, _children, next_ctx, node_key)
end

"Minimal path container for conditional/importance-weighted passes."
    struct Trajectory{Ω}
    ω::Vector{Ω}            # disturbances along the path
    w::Float64              # path weight (importance weight or 1/S)
    meta::Dict{Symbol,Any}  # optional tags (e.g., :regime => :dry, :rare => true)
end

# Keep it simple: only 1 type parameter, and fields typed to AbstractMatrix{T}
struct BaseStageData{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
    G::AbstractMatrix{T}
end

# Converting outer constructor: promotes and converts all matrices to a common eltype T
function BaseStageData(A::AbstractMatrix, B::AbstractMatrix, G::AbstractMatrix)
    T = promote_type(eltype(A), eltype(B), eltype(G))
    return BaseStageData{T}(
        convert(AbstractMatrix{T}, A),
        convert(AbstractMatrix{T}, B),
        convert(AbstractMatrix{T}, G),
    )
end

# Scenario payload with the *same single type parameter T*
struct OmegaRef{T}
    base::BaseStageData{T}
    c_u::Vector{T}
    c_x::Vector{T}
    d::Vector{T}
    h::Vector{T}
end

# Positional converting constructor: promotes all pieces to a common T
function OmegaRef(base::BaseStageData, c_u::AbstractVector, c_x::AbstractVector,
                  d::AbstractVector, h::AbstractVector)
    T = promote_type(eltype(base.A), eltype(c_u), eltype(c_x), eltype(d), eltype(h))
    baseT = BaseStageData(
        convert(AbstractMatrix{T}, base.A),
        convert(AbstractMatrix{T}, base.B),
        convert(AbstractMatrix{T}, base.G),
    )
    return OmegaRef{T}(
        baseT,
        convert(Vector{T}, c_u),
        convert(Vector{T}, c_x),
        convert(Vector{T}, d),
        convert(Vector{T}, h),
    )
end

# Nice keyword convenience (calls the positional one)
OmegaRef(; base::BaseStageData, c_u, c_x, d, h) = OmegaRef(base, c_u, c_x, d, h)



struct OmegaStageData{T}
    c_u::Vector{T}
    c_x::Vector{T}
    A::SparseMatrixCSC{T,Int}
    B::SparseMatrixCSC{T,Int}
    d::Vector{T}
    G::SparseMatrixCSC{T,Int}
    h::Vector{T}
end



"Algorithm container."
mutable struct SDDP
    stages::Vector{Stage}
    V::Vector{ValueFn{Float64}}              # V[1..T]
    T::Int
    γ::Float64
end

function SDDP(stages::Vector{Stage}; discount::Float64=1.0)
    T = length(stages)
    V = [ValueFn{Float64}() for _ in 1:T]
    SDDP(stages, V, T, discount)
end


"""
    run_sddp!(m::SDDP;
        x0::AbstractVector,
        ctx0=nothing,
        mode::Symbol = :online,
        trajectories = nothing,
        S::Int = 1,
        max_iter::Int = 1_000,
        patience::Int = 20,
        value_tol::Float64 = 0.0,
        evaluate_index::Int = 1,
        rng = Random.default_rng(),
        force_every::Int = 10,
        cut_atol::Float64 = 1e-8,
        logfn = nothing,
    ) -> NamedTuple

General SDDP training loop.

- Works with your `Stage` API (sampling via `Stage.sampler` and expectations via `Stage.children`).
- Two modes:
  - `:online` (default): sample ω on-the-fly (`forward_pass_online!`) and build **expected** cuts per visited node (`backward_pass_expected!`).
  - `:tree`: use a provided `trajectories::Vector{Trajectory}` and call `forward_pass!` + `backward_pass!`.

Stopping rules:
- `patience`: stop if **no new cuts** for this many consecutive iterations.
- `value_tol`: optional absolute tolerance on change of `V_{evaluate_index}(x0)`; set `≤ 0` to disable.

Keyword args:
- `S`: rollouts per iteration in `:online` mode.
- `force_every`: even if a node looks inactive, add an “activity-forced” cut every `force_every` iterations
  (your `backward_pass_expected!` already supports this).
- `cut_atol`: passed through to backward pass as a numerical tolerance for cut dominance/equality.
- `logfn`: optional callback `logfn(it, stats::NamedTuple)` called each iteration.

Returns a `NamedTuple` with fields:
- `iters`, `cuts_per_stage`, `history` (per-iteration stats vector).
"""
function run_sddp!(m::SDDP;
    x0::AbstractVector,
    ctx0=nothing,
    mode::Symbol = :online,
    trajectories = nothing,
    S::Int = 1,
    max_iter::Int = 1_000,
    patience::Int = 20,
    value_tol::Float64 = 0.0,
    evaluate_index::Int = 1,
    rng = Random.default_rng(),
    force_every::Int = 10,
    cut_atol::Float64 = 1e-8,
    logfn = nothing,
)

    # small helpers
    total_cuts() = sum(length(m.V[t].cuts) for t in 1:m.T)
    cuts_by_stage() = [length(m.V[t].cuts) for t in 1:m.T]

    # initial monitors
    prev_total = total_cuts()
    stagnant   = 0
    prev_V, _  = evaluate(m.V[evaluate_index], x0)

    # collect iteration stats for the caller
    hist = Vector{NamedTuple}()

    # ensure RNG is set for sampling (user can pass their own rng)
    Random.seed!(rng, rand(UInt))  # ensures independence across separate runs if user wants

    for it in 1:max_iter
        # -------- Forward --------
        fwd = if mode === :online
            forward_pass_online!(m; S=S, x0=x0, ctx0=ctx0)
        elseif mode === :tree
            trajectories === nothing &&
                error("run_sddp!: `mode=:tree` requires `trajectories` kwarg.")
            forward_pass!(m; trajectories=trajectories, x0=x0, ctx0=ctx0)
        else
            error("run_sddp!: unknown mode = $(mode). Use :online or :tree.")
        end

        # -------- Backward --------
        if mode === :online
            # one expected cut per visited node using (Xi, pXi)
            backward_pass_expected!(m; fwd=fwd, iter=it, force_every=force_every, atol=cut_atol)
        else
            backward_pass!(m; trajectories=trajectories, fwd=fwd)
        end

        # -------- Logging & stopping --------
        cur_total = total_cuts()
        new_cuts  = cur_total - prev_total
        prev_total = cur_total

        cur_V, _ = evaluate(m.V[evaluate_index], x0)
        ΔV       = abs(cur_V - prev_V)
        prev_V   = cur_V

        stats = (iter=it, new_cuts=new_cuts, total_cuts=cur_total,
                 V=cur_V, ΔV=ΔV, per_stage=cuts_by_stage())
        push!(hist, stats)

        # default console log (if no logfn provided)
        if logfn === nothing
            @printf "iter %4d | new cuts: %2d | total: %3d | V%d(x0)=%.6f | ΔV=%.3e\n" it new_cuts cur_total evaluate_index cur_V ΔV
        else
            logfn(it, stats)
        end

        stagnant = (new_cuts == 0) ? (stagnant + 1) : 0
        if stagnant >= patience
            println("Early stop: no new cuts for $patience consecutive iterations.")
            return (iters=it, cuts_per_stage=cuts_by_stage(), history=hist)
        end
        if value_tol > 0 && ΔV ≤ value_tol
            println("Early stop: |ΔV| ≤ $value_tol.")
            return (iters=it, cuts_per_stage=cuts_by_stage(), history=hist)
        end
    end

    println("Reached max_iter without early stop.")
    return (iters=max_iter, cuts_per_stage=cuts_by_stage(), history=hist)
end



############Markov version##################
mutable struct MarkovSDDP
    stages::Vector{Stage}
    # V[t][ctx_key] = ValueFn for stage t and Markov state ctx_key
    V::Vector{Dict{Any, ValueFn{Float64}}}
    T::Int
    γ::Float64
end

function MarkovSDDP(stages::Vector{Stage}; discount::Float64 = 1.0)
    T = length(stages)
    V = [Dict{Any, ValueFn{Float64}}() for _ in 1:T]
    MarkovSDDP(stages, V, T, discount)
end

function get_V!(m::MarkovSDDP, t::Int, ctx_key)
    dict = m.V[t]
    if !haskey(dict, ctx_key)
        dict[ctx_key] = ValueFn{Float64}()
    end
    return dict[ctx_key]
end


function run_markov_sddp!(m::MarkovSDDP;
                          x0::AbstractVector,
                          ctx0,
                          S::Int = 1,
                          max_iter::Int = 1_000,
                          patience::Int = 20,
                          value_tol::Float64 = 0.0,
                          evaluate_stage::Int = 1,
                          evaluate_ctx,
                          rng = Random.default_rng(),
                          force_every::Int = 10,
                          cut_atol::Float64 = 1e-8,
                          logfn = nothing)

    # Helpers that count cuts across all Markov states at each stage
    # Helpers that count cuts across all Markov states at each stage
    total_cuts() = sum(
        t -> sum(vf -> length(vf.cuts), values(m.V[t]); init = 0),
        1:m.T;
        init = 0,
    )

    cuts_by_stage() = [
        sum(vf -> length(vf.cuts), values(m.V[t]); init = 0)
        for t in 1:m.T
    ]

    prev_total = total_cuts()
    stagnant   = 0

    # initial value at evaluation (stage, ctx)
    vf_eval = get_V!(m, evaluate_stage, evaluate_ctx)
    prev_V, _ = evaluate(vf_eval, x0)

    hist = Vector{NamedTuple}()

    Random.seed!(rng, rand(UInt))

    for it in 1:max_iter
        # -------- Forward pass (Markov-aware) --------
        fwd = forward_pass_markov_online!(m; S=S, x0=x0, ctx0=ctx0)

        # -------- Backward pass (Markov expected cuts) --------
        backward_pass_markov_expected!(m; fwd=fwd, iter=it,
                                       force_every=force_every, atol=cut_atol)

        # -------- Logging & stopping --------
        cur_total = total_cuts()
        new_cuts  = cur_total - prev_total
        prev_total = cur_total

        vf_eval = get_V!(m, evaluate_stage, evaluate_ctx)
        cur_V, _ = evaluate(vf_eval, x0)
        ΔV       = abs(cur_V - prev_V)
        prev_V   = cur_V

        stats = (iter=it, new_cuts=new_cuts, total_cuts=cur_total,
                 V=cur_V, ΔV=ΔV, per_stage=cuts_by_stage())
        push!(hist, stats)

        if logfn === nothing
            @printf "iter %4d | new cuts: %2d | total: %3d | V%d[%s](x0)=%.6f | ΔV=%.3e\n" it new_cuts cur_total evaluate_stage string(evaluate_ctx) cur_V ΔV
        else
            logfn(it, stats)
        end

        stagnant = (new_cuts == 0) ? (stagnant + 1) : 0
        if stagnant >= patience
            println("Early stop: no new cuts for $patience consecutive iterations.")
            return (iters=it, cuts_per_stage=cuts_by_stage(), history=hist)
        end
        if value_tol > 0 && ΔV ≤ value_tol
            println("Early stop: |ΔV| ≤ $value_tol.")
            return (iters=it, cuts_per_stage=cuts_by_stage(), history=hist)
        end
    end

    println("Reached max_iter without early stop.")
    return (iters=max_iter, cuts_per_stage=cuts_by_stage(), history=hist)
end


#######Risk averse###########
"""
    run_markov_sddp_rho!(m::MarkovSDDP;
                         x0::AbstractVector,
                         ctx0,
                         S::Int = 1,
                         max_iter::Int = 1_000,
                         patience::Int = 20,
                         value_tol::Float64 = 0.0,
                         evaluate_stage::Int = 1,
                         evaluate_ctx,
                         rng = Random.default_rng(),
                         force_every::Int = 10,
                         cut_atol::Float64 = 1e-8,
                         alpha::Float64,
                         lambda::Float64,
                         logfn = nothing)

Train a **Markov SDDP model** under the combined coherent risk measure


    rho_{alpha,lambda}(Z) =  lambda,{E}[Z] + (1-lambda),AVaR_alpha(Z),


applied stagewise to the scenario values generated by the stage builders.

Keyword arguments:

  * `x0`          – initial state vector at stage 1.
  * `ctx0`        – initial Markov state (node key) at stage 1.
  * `S`           – number of forward trajectories per iteration.
  * `max_iter`    – maximum number of SDDP iterations.
  * `patience`    – stop if no new cuts are added for this many consecutive iterations.
  * `value_tol`   – optional tolerance on |ΔV| at the evaluation (stage,ctx); ≤ 0 disables.
  * `evaluate_stage` – stage index at which to monitor convergence (typically 1).
  * `evaluate_ctx`   – Markov state key at which to monitor convergence.
  * `rng`         – RNG object used for sampling in the forward pass.
  * `force_every` – every `force_every` iterations, force addition of a cut at each visited node
                    (even if inactive at the current support point); ≤ 0 disables forcing.
  * `cut_atol`    – numerical tolerance passed to the backward pass for cut activity/dominance.
  * `alpha`       – AVaR level α ∈ [0,1); α close to 1 focuses more on tail scenarios.
  * `lambda`      – mixing parameter λ ∈ [0,1]; λ = 1 is risk–neutral, λ = 0 is pure AVaR.
  * `logfn`       – optional callback `logfn(it, stats::NamedTuple)`; if `nothing`, a default
                    textual log is printed each iteration.

The algorithm repeatedly:

  1. Runs a Markov–aware forward pass `forward_pass_markov_online!` with `S` trajectories.
  2. For each visited node (t, ctx) and support point xₜ, performs a risk-averse backward pass
     `backward_pass_markov_rho!`, which:
       * builds scenario cuts (α_ω, β_ω) for each child shock ω,
       * evaluates scenario values v_ω = α_ω + β_ωᵀ xₜ,
       * computes optimal AVaR dual weights via `_avar_dual_weights(ps, v_ω; alpha)`,
       * forms combined weights w_ω = λ p_ω + (1−λ) λ_ω^{AVaR},
       * aggregates a single cut using these weights and adds it to the appropriate
         `ValueFn{Float64}` at stage t and Markov state ctx (if active).

Stopping rules:

  * **Cut stagnation:** terminate early if no new cuts are added for `patience` consecutive iterations.
  * **Value stabilization:** if `value_tol > 0` and the change in the monitored value
    |ΔV| ≤ `value_tol`, terminate early.

Returns a `NamedTuple` with fields:

  * `iters`           – number of iterations performed.
  * `cuts_per_stage`  – vector with the total number of cuts at each stage (aggregated over Markov states).
  * `history`         – vector of per–iteration statistics,
                        each `stats` having fields `(iter, new_cuts, total_cuts, V, ΔV, per_stage)`.
"""
function run_markov_sddp_rho!(m::MarkovSDDP;
                              x0::AbstractVector,
                              ctx0,
                              S::Int = 1,
                              max_iter::Int = 1_000,
                              patience::Int = 20,
                              value_tol::Float64 = 0.0,
                              evaluate_stage::Int = 1,
                              evaluate_ctx,
                              rng = Random.default_rng(),
                              force_every::Int = 10,
                              cut_atol::Float64 = 1e-8,
                              alpha::Float64,
                              lambda::Float64,
                              logfn = nothing)

    @assert 0.0 ≤ lambda ≤ 1.0 "lambda must be in [0,1]."
    @assert 0.0 ≤ alpha  < 1.0 "alpha must be in [0,1)."

    # Helpers that count cuts across all Markov states at each stage
    total_cuts() = sum(
        t -> sum(vf -> length(vf.cuts), values(m.V[t]); init = 0),
        1:m.T;
        init = 0,
    )

    cuts_by_stage() = [
        sum(vf -> length(vf.cuts), values(m.V[t]); init = 0)
        for t in 1:m.T
    ]

    prev_total = total_cuts()
    stagnant   = 0

    # initial value at evaluation (stage, ctx)
    vf_eval = get_V!(m, evaluate_stage, evaluate_ctx)
    prev_V, _ = evaluate(vf_eval, x0)

    # For relative improvement check over a iteration window
    const VALUE_CHECK_WINDOW = 200
    last_check_V = prev_V  # V at last value_tol check

    hist = Vector{NamedTuple}()

    Random.seed!(rng, rand(UInt))

    for it in 1:max_iter
        # -------- Forward pass (Markov-aware) --------
        fwd = forward_pass_markov_online!(m; S=S, x0=x0, ctx0=ctx0)

        # -------- Backward pass with ρ_{α,λ} --------
        backward_pass_markov_rho!(m;
            fwd         = fwd,
            iter        = it,
            force_every = force_every,
            atol        = cut_atol,
            alpha       = alpha,
            lambda      = lambda,
        )

        # -------- Logging & stopping --------
        cur_total = total_cuts()
        new_cuts  = cur_total - prev_total
        prev_total = cur_total

        vf_eval = get_V!(m, evaluate_stage, evaluate_ctx)
        cur_V, _ = evaluate(vf_eval, x0)
        ΔV       = abs(cur_V - prev_V)
        prev_V   = cur_V

        stats = (iter=it, new_cuts=new_cuts, total_cuts=cur_total,
                 V=cur_V, ΔV=ΔV, per_stage=cuts_by_stage())
        push!(hist, stats)

        if logfn === nothing
            @printf "iter %4d | new cuts: %2d | total: %3d | ρ_{α,λ} V%d[%s](x0)=%.6f | ΔV=%.3e\n" it new_cuts cur_total evaluate_stage string(evaluate_ctx) cur_V ΔV
        else
            logfn(it, stats)
        end

        stagnant = (new_cuts == 0) ? (stagnant + 1) : 0
        if stagnant >= patience
            println("Early stop: no new cuts for $patience consecutive iterations.")
            return (iters=it, cuts_per_stage=cuts_by_stage(), history=hist)
        end


        # Value-based early stop:
        # Check relative improvement only every VALUE_CHECK_WINDOW iterations
        if value_tol > 0.0 && it % VALUE_CHECK_WINDOW == 0
            denom = max(1.0, abs(last_check_V))
            rel_ΔV = abs(cur_V - last_check_V) / denom

            if rel_ΔV ≤ value_tol
                println("Early stop: relative |ΔV| over last $VALUE_CHECK_WINDOW iterations ≤ $value_tol.")
                return (iters=it, cuts_per_stage=cuts_by_stage(), history=hist)
            end

            # Reset baseline for next window
            last_check_V = cur_V
        end
    end

    println("Reached max_iter without early stop.")
    return (iters=max_iter, cuts_per_stage=cuts_by_stage(), history=hist)
end


