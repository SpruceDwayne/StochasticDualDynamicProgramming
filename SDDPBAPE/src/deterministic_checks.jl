using JuMP, HiGHS
import MathOptInterface as MOI

"""
    solve_extensive_control(m::SDDP;
        B0::AbstractVector,
        c0::AbstractVector,
        ucap0::Union{Real,AbstractVector}=Inf,
        ucap::Union{Real,AbstractVector}=Inf,
        stage_cost = (t, ω) -> ω,
        dynamics   = (t, ω) -> (Matrix{Float64}(I, m.stages[t].state_dim, m.stages[t].state_dim),
                                 -I,
                                 zeros(m.stages[t].state_dim)),
        Aeq_beq::Union{Nothing,Function}=nothing,
        Ale_ble::Union{Nothing,Function}=nothing,
        ctx0 = nothing,
    ) -> (x0_star::Vector{Float64}, obj::Float64, model::Model)

Build and solve the **deterministic-equivalent (extensive-form) LP** for any SDDP model
that follows the *control template*:

- State (vector) `B_t ∈ ℝⁿ`.
- Control (vector) `u_t ∈ ℝⁿ`, default nonnegative with optional componentwise caps.
- Linear dynamics per node: `B_{t+1} = A_t(ω) * B_t + D_t(ω) * u_t + b_t(ω)`.
- Linear stage cost on the control:
    - At stage 0 (here-and-now): cost is `- dot(c0, u0)`.
    - At stages `t ≥ 1`: cost is `- dot(c_t(ω), u_t)` with `c_t` provided by `stage_cost`.

The scenario tree is expanded **exactly** from the stage API
`children(t, ctx)::(Ωs, ps)` and `next_ctx(t, ctx, ω)`. The objective is the **expected**
sum of stage costs. The model enforces nonanticipativity by having one decision per tree node.

# Arguments
- `m::SDDP`:
    Your SDDP container (`stages::Vector{Stage}`, `T`, …). Only `children`, `next_ctx`,
    and `stages[t].state_dim` are used here.
- `B0::AbstractVector`:
    Initial state at stage 0; length defines the state and control dimension `n`.
- `c0::AbstractVector`:
    Time-0 cost coefficients for `u0` (same length as `B0`).
- `ucap0`, `ucap`:
    Componentwise **upper bounds** on `u_t`. May be a scalar (broadcasts to all components)
    or a vector of length `n`. Use `Inf` (or vector of `Inf`) for “no cap beyond nonnegativity.”
- `stage_cost::Function` (optional):
    `(t, ω) -> c`. Returns either a **scalar** (applied to `sum(u_t)`) or a **vector** of
    length `n` (applied to `dot(c, u_t)`). Nodes with `ω === nothing` (e.g., deterministic
    pass-through children from a stage-0 wrapper) are **skipped** in the stochastic sum;
    stage-0 cost is handled by `c0`.
    Default: `ω` (scalar), which matches the resource example `cost = -ω * u_t`.
- `dynamics::Function` (optional):
    `(t, ω) -> (A, D, b)` returning matrices/vectors sized for the state/control dimension,
    implementing `B_{t+1} = A B_t + D u_t + b`. The **default** is
    `A = I`, `D = -I`, `b = 0`, i.e., `B_{t+1} = B_t - u_t`.
- `Aeq_beq::Function` (optional):
    `(t, ω) -> (Aeq, beq)` to impose **per-node equalities** `Aeq * u_t == beq`.
    Return `(nothing, nothing)` when not applicable.
- `Ale_ble::Function` (optional):
    `(t, ω) -> (Ale, ble)` to impose **per-node inequalities** `Ale * u_t ≤ ble`.
    Return `(nothing, nothing)` when not applicable.
- `ctx0` (optional):
    Initial node context. Defaults to `nothing`.

# Returns
- `x0_star::Vector{Float64}`: Optimal here-and-now control `u0` (respects `ucap0`).
- `obj::Float64`           : Optimal extensive-form expected objective value.
- `model::Model`           : The JuMP model (inspect `value.(u[i, :])`, `value.(B[i, :])` etc.).

# Notes
- Controls are **nonnegative** by construction. To allow signed controls, change the
  `@variable(model, u[1:N, 1:n] >= 0)` line accordingly.
- The tree is exact (no sampling). This is intended as a **gold-standard** checker for
  small to medium trees.
- The function is **dimension-agnostic** (works for `n > 1`) as long as you provide
  compatible `c0`, `stage_cost`, and `dynamics`.
- Deterministic stage-0 wrappers that produce a child with `ω === nothing` are supported;
  their contribution to cost is already accounted for via `-dot(c0, u0)`.

# Examples

Minimal use for the resource planning toy (n = 1, `B_{t+1} = B_t - u_t`, scalar ROI):
```julia
x0, obj, ef = solve_extensive_control(m;
    B0     = [100.0],
    c0     = [1.06],
    ucap0  = 40.0,
    ucap   = 40.0,
    # stage_cost, dynamics left as defaults
)
println("x0* = ", x0[1], ", obj = ", obj)
"""
# ===================== General deterministic-equivalent solver =====================
using JuMP, HiGHS
import MathOptInterface as MOI

function solve_extensive_control(m::SDDP;
    B0::AbstractVector,
    c0::AbstractVector,
    ucap0::Union{Real,AbstractVector}=Inf,
    ucap::Union{Real,AbstractVector}=Inf,
    stage_cost = (t,ω)->ω,  # may return scalar or vector cost on u_t; ω may be `nothing` at deterministic nodes
    dynamics   = (t,ω)->(Matrix{Float64}(I, m.stages[t].state_dim, m.stages[t].state_dim),
                          -I,
                          zeros(m.stages[t].state_dim)),
    Aeq_beq::Union{Nothing,Function}=nothing,     # (t,ω) -> (Aeq, beq) or (nothing, nothing)
    Ale_ble::Union{Nothing,Function}=nothing,     # (t,ω) -> (Ale, ble) or (nothing, nothing)
    ctx0 = nothing,
)
    T = m.T
    n = length(B0)

    # ---- expand exact tree from Stage API ----
    nodes = NamedTuple[]  # (id, t, parent, prob, ctx, ω)
    push!(nodes, (id=1, t=0, parent=0, prob=1.0, ctx=ctx0, ω=nothing))
    level_ids = Vector{Vector{Int}}(undef, T+1)
    level_ids[1] = [1]
    next_id = 2
    for t in 1:T
        level_ids[t+1] = Int[]
        stg = m.stages[t]
        for nid in level_ids[t]
            parent = nodes[nid]
            Ωs, ps = stg.children(t, parent.ctx)
            @assert length(Ωs) == length(ps)
            @assert abs(sum(ps) - 1.0) < 1e-10
            for (ω, p) in zip(Ωs, ps)
                ctx′ = stg.next_ctx(t, parent.ctx, ω)
                push!(nodes, (id=next_id, t=t, parent=nid,
                              prob=parent.prob * p, ctx=ctx′, ω=ω))
                push!(level_ids[t+1], next_id)
                next_id += 1
            end
        end
    end
    N = length(nodes)

    # children map
    children = Dict{Int, Vector{Int}}(i=>Int[] for i in 1:N)
    for n in nodes
        if n.parent != 0
            push!(children[n.parent], n.id)
        end
    end

    # ---- helpers ----
    to_vec(v, dim) = (v isa AbstractVector ? collect(Float64.(v)) :
                      isfinite(v) ? fill(Float64(v), dim) : fill(Inf, dim))

    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, B[1:N, 1:n])
    @variable(model, u[1:N, 1:n] >= 0)

    # stage-0 feasibility
    @constraint(model, B[1, :] .== B0)
    @constraint(model, u[1, :] .<= B[1, :])
    cap0 = to_vec(ucap0, n)
    for j in 1:n
        if isfinite(cap0[j]); @constraint(model, u[1,j] <= cap0[j]); end
    end

    # stages ≥1 feasibility
    cap = to_vec(ucap, n)
    for i in 2:N
        @constraint(model, u[i, :] .<= B[i, :])
        for j in 1:n
            if isfinite(cap[j]); @constraint(model, u[i,j] <= cap[j]); end
        end
    end

    # dynamics: B_child = A B_parent + D u_parent + b
    for i in 1:N
        t = nodes[i].t
        for c in children[i]
            # child is at stage t+1 with its ω
            A, D, b = dynamics(t+1, nodes[c].ω)
            @constraint(model, B[c, :] .== (A * B[i, :]) + (D * u[i, :]) + b)
        end
    end

    # optional extra constraints on u_t
    # stage 0
    if Aeq_beq !== nothing
        Aeq0, beq0 = Aeq_beq(0, nothing)
        if Aeq0 !== nothing; @constraint(model, Aeq0 * u[1, :] .== beq0); end
    end
    if Ale_ble !== nothing
        Ale0, ble0 = Ale_ble(0, nothing)
        if Ale0 !== nothing; @constraint(model, Ale0 * u[1, :] .<= ble0); end
    end
    # stages ≥1
    for i in 2:N
        t = nodes[i].t; ω = nodes[i].ω
        if Aeq_beq !== nothing
            Aeq, beq = Aeq_beq(t, ω)
            if Aeq !== nothing; @constraint(model, Aeq * u[i, :] .== beq); end
        end
        if Ale_ble !== nothing
            Ale, ble = Ale_ble(t, ω)
            if Ale !== nothing; @constraint(model, Ale * u[i, :] .<= ble); end
        end
    end

    # objective: stage-0 cost + expected future costs (skip ω===nothing)
    @expression(model, obj0, -dot(c0, u[1, :]))
    @expression(model, obj_future,
        sum( begin
                c = stage_cost(nodes[i].t, nodes[i].ω)
                if c isa Number
                    -c * sum(u[i, :]) * nodes[i].prob
                else
                    -dot(c, u[i, :]) * nodes[i].prob
                end
             end for i in 2:N if nodes[i].ω !== nothing)
    )
    @objective(model, Min, obj0 + obj_future)

    optimize!(model)
    term = termination_status(model)
    if !(term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED))
        error("Deterministic-equivalent solve not optimal: $term")
    end
    return value.(u[1, :]), objective_value(model), model
end



