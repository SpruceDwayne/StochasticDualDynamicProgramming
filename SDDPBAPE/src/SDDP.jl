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
        if abs(c.α - α) <= atol && norm(c.β - β) <= atol
            return false
        end
    end
    push!(v.cuts, Cut{T}(α, β, stage))
    return true
end

"Stage definition: user supplies a builder to create the JuMP model for a given ω."
struct Stage
    t::Int
    state_dim::Int
    build::Function   # (t, vf_next, ω; fix_state=nothing) -> (model, x_state, θ, misc)
    sampler::Function # () -> ω
    weight::Function  # ω -> probability/weight
end

"Scenario tree + trajectories (optional, if you want deterministic trees/sampling)."
struct TreeNode{Ω}
    id::Int
    parent::Union{Nothing,Int}
    ω::Ω
    p_cond::Float64
end
struct ScenarioTree{Ω}
    levels::Vector{Vector{TreeNode{Ω}}}
end
struct Trajectory{Ω}
    node_ids::Vector{Int}
    ω::Vector{Ω}
    p::Float64
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


# Enumerate all root→leaf trajectories
function enumerate_trajectories(tree::ScenarioTree{Ω}) where {Ω}
    T = length(tree.levels)
    results = Trajectory{Ω}[]
    nodebuf = Vector{Int}(undef, T)
    ωbuf    = Vector{Ω}(undef, T)

    function rec(t::Int, parent_id::Union{Nothing,Int}, p::Float64)
        if t == 1
            for n in tree.levels[1]
                nodebuf[1] = n.id
                ωbuf[1]    = n.ω
                rec(2, n.id, p * n.p_cond)
            end
            return
        end
        if t > T
            push!(results, Trajectory(copy(nodebuf), copy(ωbuf), p))
            return
        end
        for n in tree.levels[t]
            if n.parent == parent_id
                nodebuf[t] = n.id
                ωbuf[t]    = n.ω
                rec(t+1, n.id, p * n.p_cond)
            end
        end
    end

    rec(1, nothing, 1.0)
    return results
end

# Sample n trajectories IID from the tree distribution
function sample_trajectories(tree::ScenarioTree{Ω}, n_samples::Int) where {Ω}
    T   = length(tree.levels)
    out = Vector{Trajectory{Ω}}(undef, n_samples)

    for s in 1:n_samples
        node_ids = Vector{Int}(undef, T)
        ω        = Vector{Ω}(undef, T)
        p        = 1.0

        # root
        r = rand(); acc = 0.0
        root = nothing
        for n in tree.levels[1]
            acc += n.p_cond
            if r <= acc
                root = n; break
            end
        end
        @assert root !== nothing "root sampling failed (check root p_cond sums to 1)"

        node_ids[1] = root.id
        ω[1]        = root.ω
        p          *= root.p_cond
        parent      = root.id

        # descend
        for t in 2:T
            children = (n for n in tree.levels[t] if n.parent == parent)
            r = rand(); acc = 0.0
            chosen = nothing
            for n in children
                acc += n.p_cond
                if r <= acc
                    chosen = n; break
                end
            end
            @assert chosen !== nothing "sampling failed at t=$t (check child p_cond sums to 1)"

            node_ids[t] = chosen.id
            ω[t]        = chosen.ω
            p          *= chosen.p_cond
            parent      = chosen.id
        end

        out[s] = Trajectory(node_ids, ω, p)
    end

    return out
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

function check_omega(ω::OmegaRef{T}, n::Int, m::Int) where T
    @assert size(ω.base.A) == (n,n)
    @assert size(ω.base.B) == (n,m)
    @assert size(ω.base.G,2) == m
    @assert length(ω.d) == n
    @assert length(ω.c_x) == n
    @assert length(ω.c_u) == m
    @assert length(ω.h) == size(ω.base.G,1)
end


