using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
#Pkg.instantiate()   # only needed the first time on a new checkout
using SDDPBAPE

using JuMP, HiGHS, LinearAlgebra
using Printf

# -----------------------
# Problem setup
# -----------------------
const T  = 4
const n  = 1                # state dim: B_t
const γ  = 1.0
B0 = [100.0]                # initial budget

# Discrete ROI (i.i.d. across t)
const RL, RM, RH = 0.5, 1.0, 2.0
const pL, pM, pH = 0.3, 0.5, 0.2
const Xi  = [RL, RM, RH]
const pXi = [pL, pM, pH]


# Enforce B_{T+1} = 0?
const HARD_TERMINAL = false

# -----------------------
# Stage builder (linear, uses only ω)
# -----------------------

function build_stage_model(t::Int, vf_next::ValueFn{Float64}, ω::Float64;
                           fix_state::Union{Nothing,Vector{Float64}}=nothing,
                           ucap::Float64 = Inf)   # NEW: absolute cap
    model = Model(HiGHS.Optimizer); set_silent(model)
    n = 1  # your toy

    @variable(model, x_state[1:n])
    @variable(model, x_next[1:n] >= 0)
    @variable(model, 0 <= x)                       # allocation
    @variable(model, θ)

    fix_state === nothing && error("builder needs fix_state=x_t")
    @constraint(model, x_state .== fix_state)

    # Feasibility
    @constraint(model, x <= x_state[1])            # cannot spend more than you have
    if isfinite(ucap)
        @constraint(model, x <= ucap)              # NEW: absolute cap
    end

    # Dynamics
    eq = @constraint(model, x_next[1] == x_state[1] - x)

    # Epigraph of V_{t+1}
    if isempty(vf_next.cuts)
        @constraint(model, θ >= 0)
    else
        for c in vf_next.cuts
            @constraint(model, θ >= c.α + c.β[1]*x_next[1])
        end
    end

    # Objective (min form)
    @objective(model, Min, -ω * x + θ)

    misc = Dict(
        :x_state  => x_state,
        :x_next   => x_next,
        :xnext_eq => [eq],
        :c_x      => [0.0],
        :u        => x,
    )
    return model, x_state, θ, misc
end


# -----------------------
# Build stages WITHOUT a tree
#   - sampler(): draws ω ~ Xi with probs pXi
#   - children(): returns (Xi, pXi) for exact expectation in backward pass
#   - next_ctx(): no context needed (returns nothing)
#   - node_key(): groups all visits at stage t (ctx=nothing)
# -----------------------

# If you kept the helper Stage_constant_Xi in your package, this is the cleanest:
have_helper = isdefined(SDDPBAPE, :Stage_constant_Xi)

ucap = 40.0  # e.g., can spend at most 30 per stage

stages = [
    Stage(
        t, 1,
        (tt, vf_next, ω; fix_state=nothing)->build_stage_model(tt, vf_next, ω; fix_state, ucap=ucap),
        ()->begin r=rand(); r<0.3 ? 0.5 : (r<0.8 ? 1.0 : 2.0) end,
        ω->(ω==0.5 ? 0.3 : (ω==1.0 ? 0.5 : 0.2)),
        (tt, _ctx)->(Float64[0.5,1.0,2.0], [0.3,0.5,0.2]),
        (tt, ctx, ω)->ctx,
        x->x
    )
    for t in 1:T
]
#########Building stage 0 with fixed ROI#####################

# ---------- Time-0 builder (deterministic) with ABSOLUTE cap only ----------
function build_stage0(_t::Int, vf_next::ValueFn{Float64}, _ω=nothing;
                      fix_state::Vector{Float64},
                      c0::Float64,
                      ucap0::Float64 = Inf)

    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, x_state[1:1])
    @variable(model, 0 <= u0)                   # here-and-now decision
    @variable(model, x_next[1:1] >= 0)
    @variable(model, θ)

    @constraint(model, x_state .== fix_state)
    @constraint(model, u0 <= x_state[1])
    if isfinite(ucap0)
        @constraint(model, u0 <= ucap0)
    end
    eq = @constraint(model, x_next[1] == x_state[1] - u0)

    if isempty(vf_next.cuts)
        @constraint(model, θ >= 0)
    else
        for c in vf_next.cuts
            @constraint(model, θ >= c.α + c.β[1] * x_next[1])
        end
    end
    @objective(model, Min, -c0 * u0 + θ)

    
    misc = Dict{Symbol,Any}(
        :x_state  => x_state,
        :x_next   => x_next,
        :xnext_eq => [eq],
        :c_x      => [0.0],
        :u        => u0,          
    )
    return model, x_state, θ, misc
end

# ---------- Deterministic Stage helper (positional constructor wrapper) ----------
Stage_deterministic(; t, state_dim, build, node_key = _->:root) = Stage(
    t,
    state_dim,
    build,
    ()->nothing,                          # sampler
    _->1.0,                               # weight
    (tt, _ctx)->(Any[nothing], [1.0]),    # children
    (tt, ctx, _ω)->ctx,                   # next_ctx
    node_key
)

# ---------- Build stage 0 and prepend it ----------
c0 = 1.06
ucap0 = ucap+0.0      

stage0 = Stage_deterministic(
    t = 1,
    state_dim = 1,
    build = (tt, vf_next, ω; fix_state=nothing) -> begin
        fix_state === nothing && error("stage 0 needs fix_state = B0")
        build_stage0(tt, vf_next, ω; fix_state=fix_state, c0=c0, ucap0=ucap0)
    end,
    node_key = _->:root
)

# Prepend stage0 to your existing stochastic stages
stages0_T = vcat([stage0], stages)

# IMPORTANT: construct SDDP with stages0_T, not stages
m = SDDP(stages0_T; discount=γ)


using Random
Random.seed!(1234)  # reproducible sampling (optional)

"""
Train SDDP until cuts stagnate (or max_iter reached).

Arguments
- m::SDDP                      : your SDDP container
- x0::Vector{Float64}         : initial state (e.g., B0)
- S::Int                      : rollouts per iteration (default 1)
- max_iter::Int               : max outer iterations (default 1000)
- patience::Int               : stop if no new cuts for this many consecutive iters (default 20)
- value_tol::Float64          : optional absolute tolerance on V1(x0) change (set ≤ 0 to disable)

Returns
- (iters_done, cuts_per_stage) where cuts_per_stage is a Vector{Int}
"""
function run_sddp_old!(m::SDDP; x0::Vector{Float64}, S::Int=1, max_iter::Int=1000,
                   patience::Int=20, value_tol::Float64=0.0)

    # helpers to count cuts and monitor value
    total_cuts() = sum(length(m.V[t].cuts) for t in 1:m.T)
    cuts_by_stage() = [length(m.V[t].cuts) for t in 1:m.T]

    prev_total = total_cuts()
    stagnant = 0

    prev_V1, _ = evaluate(m.V[1], x0)

    for it in 1:max_iter
        # Forward: sample ω on the fly (no explicit tree)
        fwd = forward_pass_online!(m; S=S, x0=x0, ctx0=nothing)

        # Backward: add one expected cut per visited node, using (Xi, pXi)
        backward_pass_expected!(m; fwd=fwd, iter=it, force_every=10, atol=1e-8)

        # Stagnation checks
        cur_total = total_cuts()
        new_cuts = cur_total - prev_total
        prev_total = cur_total

        cur_V1, _ = evaluate(m.V[1], x0)
        ΔV = abs(cur_V1 - prev_V1)
        prev_V1 = cur_V1

        println(@sprintf("iter %4d | new cuts: %2d | total: %3d | V1(x0)=%.6f | ΔV=%.3e",
                         it, new_cuts, cur_total, cur_V1, ΔV))

        if new_cuts == 0
            stagnant += 1
        else
            stagnant = 0
        end

        # stop if no new cuts for 'patience' iters
        if stagnant >= patience
            println("Early stop: no new cuts for $patience consecutive iterations.")
            return it, cuts_by_stage()
        end

        # optional value-convergence stop
        if value_tol > 0 && ΔV ≤ value_tol
            println("Early stop: |ΔV1(x0)| ≤ $value_tol.")
            return it, cuts_by_stage()
        end
    end

    println("Reached max_iter without triggering early stop.")
    return max_iter, cuts_by_stage()
end




iters, cuts_per_stage = run_sddp!(m; x0=B0, S=1, max_iter=1000, patience=20, value_tol=0.0)
println("Stopped after $iters iterations. Cuts per stage = ", cuts_per_stage)


######Investigate results##################

using Plots


# ---------------- Example usage ----------------
# Pick which continuation V to use:
# v_index = 1  # no explicit stage-0 in your model
# v_index = 2  # if you inserted a deterministic stage-0
v_index = 2


# Read the optimal here-and-now x0 from the deterministic stage 0
stg0 = m.stages[1]
vf1  = m.V[2]
model, x_state, θ, misc = stg0.build(1, vf1, nothing; fix_state = B0) 
optimize!(model)

x0_star = value(misc[:u])               # here-and-now spend 
B1_star = value.(misc[:x_next])         # next state
obj     = objective_value(model)

println("x0* = $x0_star,  B1* = $(B1_star[1]),  obj = $obj")


##############
v_index = 2
B1s = 0:1:100
marg = Float64[]
for b in B1s
    _, β = evaluate(m.V[v_index], [b])
    push!(marg, -β[1])
end
println("min(-β)=", minimum(marg), "  max(-β)=", maximum(marg))


###############
# inspect the time-0 objective and its “argmin set”
function time0_objective_curve(m; B0=100.0, c0=1.05, v_index=2, step=1.0)
    xs   = 0.0:step:B0
    vals = Float64[]
    for x0 in xs
        v, _ = evaluate(m.V[v_index], [B0 - x0])  # pure recourse
        push!(vals, -c0*x0 + v)
    end
    minval = minimum(vals)
    mins   = [x for (x,y) in zip(xs, vals) if isapprox(y, minval; atol=1e-7)]
    return xs, vals, minval, mins
end

xs, vals, minval, mins = time0_objective_curve(m; B0=100.0, c0=1.05, v_index=2, step=1.0)
println("min objective = $minval at x0 in ", mins)



################Check if we found the correct solution by solving DEF #########

x0_star_det, obj_det, ef_model =
    solve_extensive_control(m;
        B0 = [100.0],
        c0 = [1.06],
        ucap0 = 40.0,
        ucap = 40.0,
        # stage_cost, dynamics left as defaults for B_{t+1}=B_t - u_t
    )

println("EF says x0* = ", x0_star_det[1], ", obj = ", obj_det)
