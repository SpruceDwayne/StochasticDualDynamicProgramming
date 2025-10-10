using LinearAlgebra
using JuMP

"Forward pass artifacts."
struct ForwardRecord
    x_state::Vector{Vector{Vector{Float64}}}   # x_state[s][t] visited in forward pass
end

"""
forward_pass!(m::SDDP; trajectories, x0)

Rolls policy over given trajectories using current cuts in m.V.
(We only need the visited states for the backward pass.)
"""
function forward_pass!(m::SDDP; trajectories::Vector{Trajectory}, x0::Vector{Float64})
    T = m.T
    S = length(trajectories)
    x_hist = [ [zeros(m.stages[1].state_dim) for _ in 1:T] for _ in 1:S ]

    for (s, tr) in enumerate(trajectories)
        x = copy(x0)
        for t in 1:T
            stg = m.stages[t]
            ωt  = tr.ω[t]
            vf_next = m.V[min(t+1, T)]

            model, x_state, θ, misc = stg.build(t, vf_next, ωt; fix_state = nothing)
            optimize!(model)  # make sure your builder set a linear solver

            # advance state: prefer x_next if provided
            x = haskey(misc, :x_next) ? value.(misc[:x_next]) : value.(x_state)
            x_hist[s][t] = x
        end
    end
    return ForwardRecord(x_hist)
end

# -----------------------
# Dual-based cut from θ-epigraph + dynamics
# -----------------------

"""
Compute a stage-t cut (α, β) using ONLY duals from the solved model.

Requirements in `misc` from the builder:
  :x_state  :: Vector{VariableRef}           # the x_t variables (support point variables)
  :xnext_eq :: Vector{ConstraintRef}         # constraints x_next[i] - f_i(x,u,ω) == 0

Optional:
  :c_x      :: Vector{Float64}               # linear state cost added directly to β

Algorithm:
  - Read π = dual.(xnext_eq)  (dual for dynamics equalities)
  - For each state component j:
        β[j] = - Σ_i π[i] * coeff(x_state[j]) in xnext_eq[i]
  - If c_x present, β += c_x
  - α = objective_value(model) - β' * x_support
"""
function compute_cut!(t::Int, model, stg::Stage, vf_next::ValueFn{Float64},
                      x_support::Vector{Float64}, ω, misc)::Tuple{Float64,Vector{Float64}}

    @assert haskey(misc, :x_state)   "builder must put :x_state in misc"
    @assert haskey(misc, :xnext_eq)  "builder must put :xnext_eq in misc"

    x_state  = misc[:x_state]  ::Vector{VariableRef}
    xnext_eq = misc[:xnext_eq] ::Vector{ConstraintRef}

    π = dual.(xnext_eq)  # length = state_dim
    n = stg.state_dim
    β = zeros(n)

    # β = - (∂f/∂x)^T * π   (recovered via equality coefficients)
    for j in 1:n
        acc = 0.0
        for i in 1:n
            a_ij = JuMP.coefficient(xnext_eq[i], x_state[j])   # = -(∂f_i/∂x_j)
            acc += π[i] * a_ij
        end
        β[j] = -acc
    end

    if haskey(misc, :c_x)
        β .+= misc[:c_x]
    end

    α = objective_value(model) - dot(β, x_support)
    return (α, β)
end

# -----------------------
# Helper: bucket paths by node id at stage t (so we add ONE expected cut per node)
# -----------------------
function group_by_node(traj::Vector{Trajectory}, t::Int)
    buckets = Dict{Int, Vector{Int}}()
    for (s, tr) in enumerate(traj)
        nid = tr.node_ids[t]
        push!(get!(buckets, nid, Int[]), s)
    end
    return buckets
end

"""
backward_pass!(m::SDDP; trajectories, fwd)

Shared (node-expected) cuts:
- For each stage t and each visited node at that stage,
  compute a weighted average of cuts produced by the scenarios that visit that node.
- Weights default to the scenarios' *path probabilities normalized within the node bucket*,
  which equals the conditional child probabilities if your samples are IID from the tree.
"""
function backward_pass!(m::SDDP; trajectories::Vector{Trajectory}, fwd::ForwardRecord)
    T = m.T
    for t in T:-1:1
        vf_next = m.V[min(t+1, T)]
        stg = m.stages[t]

        buckets = group_by_node(trajectories, t)

        for (nid, scen_idx) in buckets
            # normalized weights within this node (SAA of conditional probs)
            ws = [trajectories[s].p for s in scen_idx]
            wsum = sum(ws)
            if wsum == 0.0
                ws .= 1.0
                wsum = length(ws)
            end
            ws ./= wsum

            α_acc = 0.0
            β_acc = zeros(stg.state_dim)

            for (w, s) in zip(ws, scen_idx)
                x_support = fwd.x_state[s][t]
                ωt        = trajectories[s].ω[t]

                model, x_state, θ, misc = stg.build(t, vf_next, ωt; fix_state = x_support)
                optimize!(model)

                # ensure compute_cut! can find x_state in misc even if builder didn't stash it
                misc[:x_state] = get(misc, :x_state, x_state)

                α, β = compute_cut!(t, model, stg, vf_next, x_support, ωt, misc)
                α_acc += w * α
                β_acc .+= w .* β
            end

            add_cut!(m.V[t], α_acc, β_acc, t)
        end
    end
    return nothing
end
