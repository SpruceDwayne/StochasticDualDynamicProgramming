using LinearAlgebra
using SparseArrays  # if you use SparseMatrixCSC


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



