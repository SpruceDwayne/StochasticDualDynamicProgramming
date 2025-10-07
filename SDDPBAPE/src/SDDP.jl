using JuMP
using LinearAlgebra   # for dot, norm

"Canonical node type for a scenario tree."
struct TreeNode{Ω}
    id::Int
    parent::Union{Nothing,Int}
    ω::Ω                    # shock (whatever your build expects)
    p_cond::Float64         # P(node | parent)  (root typically has p_cond = 1.0)
end

"Tree grouped by level: levels[t] = nodes at stage t. Children are those with parent==node.id."
struct ScenarioTree{Ω}
    levels::Vector{Vector{TreeNode{Ω}}}
end

"Convenience: unconditional probability of a node is product of p_cond up the path."
function node_probability(tree::ScenarioTree, t::Int, node_id::Int)
    # O(depth) walk. Cache this if the tree is big.
    id = node_id
    prob = 1.0
    for s in t:-1:2  # walk back to stage 1
        node = first(n for n in tree.levels[s] if n.id == id)
        prob *= node.p_cond
        id = node.parent::Int
    end
    # root
    prob *= 1.0
    return prob
end


"One full path through the tree."
struct Trajectory{Ω}
    node_ids::Vector{Int}   # node_ids[t] is the chosen node at stage t
    ω::Vector{Ω}            # ω[t] at each stage t
    p::Float64              # unconditional probability of this path
end

"Enumerate or sample trajectories."

"Depth-first enumeration (useful for small trees, deterministic)."
function enumerate_trajectories(tree::ScenarioTree)
    T = length(tree.levels)
    results = Trajectory[]
    buf_nodes = Vector{Int}(undef, T)
    buf_ω     = Vector{eltype(first(tree.levels)).ω}(undef, T)

    function rec(t::Int, prob::Float64, parent_id::Union{Nothing,Int})
        if t == 1
            for n in tree.levels[1]
                buf_nodes[1] = n.id
                buf_ω[1] = n.ω
                rec(2, prob * n.p_cond, n.id)
            end
            return
        end
        if t > T
            push!(results, Trajectory(copy(buf_nodes), copy(buf_ω), prob))
            return
        end
        for n in tree.levels[t]
            if n.parent == parent_id
                buf_nodes[t] = n.id
                buf_ω[t] = n.ω
                rec(t+1, prob * n.p_cond, n.id)
            end
        end
    end

    rec(1, 1.0, nothing)
    return results
end

"Random sampling of trajectories according to child conditional probabilities."
function sample_trajectories(tree::ScenarioTree, n_samples::Int)
    T = length(tree.levels)
    out = Vector{Trajectory}(undef, n_samples)
    for s in 1:n_samples
        node_ids = Vector{Int}(undef, T)
        ω        = Vector{eltype(first(tree.levels)).ω}(undef, T)
        prob = 1.0
        # pick root
        roots = tree.levels[1]
        # Their p_cond should sum to 1 (or you handle otherwise).
        r = rand()
        acc = 0.0
        root = nothing
        for n in roots
            acc += n.p_cond
            if r <= acc
                root = n; break
            end
        end
        @assert root !== nothing "No root sampled; check p_cond at t=1"
        node_ids[1] = root.id; ω[1] = root.ω; prob *= root.p_cond
        parent = root.id

        for t in 2:T
            children = [n for n in tree.levels[t] if n.parent == parent]
            r = rand()
            acc = 0.0
            chosen = nothing
            for n in children
                acc += n.p_cond
                if r <= acc
                    chosen = n; break
                end
            end
            @assert chosen !== nothing "No child sampled at t=$t; check p_cond"
            node_ids[t] = chosen.id
            ω[t] = chosen.ω
            prob *= chosen.p_cond
            parent = chosen.id
        end
        out[s] = Trajectory(node_ids, ω, prob)
    end
    return out
end















#-----------------------------------------------------
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
    build::Function   # (t, vf_next, ω; fix_state = nothing) -> (model, x_state, theta, misc)
    sampler::Function # () -> ω  (noise for this stage)
    ϕ::Function       # (ω) -> probability or weight (risk-neutral → 1/#scenarios)
end

"Algorithm container."
mutable struct SDDP
    stages::Vector{Stage}
    V::Vector{ValueFn{Float64}}              # V[1..T]
    T::Int
    γ::Float64
    trial_points::Vector{Vector{Vector{Float64}}}
end

function SDDP(stages::Vector{Stage}; discount::Float64=1.0)
    T = length(stages)
    V = [ValueFn{Float64}() for _ in 1:T]
    SDDP(stages, V, T, discount, [Vector{Vector{Float64}}() for _ in 1:T])
end

