#This script is an extended version of the problem sovled in inventory_example.jl where we now assume the uncertainty follows a makovchain instead of an IID process.
#For clarity think of the uncertainty as representing weather, and markov states as states defininng the probability of different weather types
#Eg. state = sunny preiod, then P(sunny) is higher and P(rain) is lower etc
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
Pkg.resolve()        # clears those “dependencies changed” warnings
Pkg.instantiate()  # only needed the first time on a new checkout
Pkg.precompile()
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
ucap = 40.0

# -----------------------
# Markov weather states
# -----------------------
const SUN    = 1
const CLOUDY = 2
const RAIN   = 3
const K      = 3  # number of Markov states

# Transition matrix P(z_{t+1} | z_t)
const Pz = [
    0.7  0.2  0.1;  # from SUN    -> (SUN, CLOUDY, RAIN)
    0.3  0.5  0.2;  # from CLOUDY -> ...
    0.1  0.3  0.6;  # from RAIN   -> ...
]

# ROI levels (can be shared across states or later made state-specific)
const RL, RM, RH = 0.5, 1.0, 2.0
const ROI_levels = [RL, RM, RH]

# For each z in {SUN,CLOUDY,RAIN}, specify probs for RL, RM, RH.
# (Right now the omega *set* is the same, but probabilities differ by state.
#  You can generalize to state-specific omega sets later if you like.)
const pROI = Dict(
    SUN    => [0.1, 0.3, 0.6],
    CLOUDY => [0.4, 0.4, 0.2],
    RAIN   => [0.7, 0.25, 0.05],
)


# Joint shock enumeration for given current Markov state z:
# builds the set { (z_next, roi) } with probabilities P(z_next, roi | z)
function Xi_of(z::Int)
    shocks = WeatherShock[]
    probs  = Float64[]
    for z_next in 1:K
        Pz_row = Pz[z, z_next]
        if Pz_row == 0.0
            continue
        end
        # ROI distribution conditional on *next* state z_next OR current z;
        # here we keep it conditional on current state z, as coded in pROI[z]
        rois  = ROI_levels
        prows = pROI[z]  # if you want dependence on z_next instead, use pROI[z_next]
        @assert length(rois) == length(prows)
        for (roi, p_roi) in zip(rois, prows)
            p = Pz_row * p_roi  # joint probability
            if p > 0
                push!(shocks, WeatherShock(roi, z_next))
                push!(probs, p)
            end
        end
    end
    return shocks, probs
end

# (Not strictly needed, but kept for completeness / compatibility.)
function pXi_of(z::Int)
    _, probs = Xi_of(z)
    return probs
end

# -----------------------
# Stage builder (same as in IID example)
# -----------------------
function build_stage_model(t::Int, vf_next::ValueFn{Float64}, ω::Float64;
                           fix_state::Union{Nothing,Vector{Float64}} = nothing,
                           ucap::Float64 = Inf)

    model = Model(HiGHS.Optimizer); set_silent(model)
    n = 1  # one resource: fuel/budget

    @variable(model, x_state[1:n])
    @variable(model, x_next[1:n] >= 0)
    @variable(model, 0 <= x)      # allocation / spend
    @variable(model, θ)

    fix_state === nothing && error("builder needs fix_state = x_t")
    @constraint(model, x_state .== fix_state)

    # Feasibility: cannot spend more than you have, and not above per-stage cap
    @constraint(model, x <= x_state[1])
    if isfinite(ucap)
        @constraint(model, x <= ucap)
    end

    # Dynamics: B_{t+1} = B_t - x
    eq = @constraint(model, x_next[1] == x_state[1] - x)

    # Epigraph of V_{t+1}
    if isempty(vf_next.cuts)
        @constraint(model, θ >= 0)
    else
        for c in vf_next.cuts
            @constraint(model, θ >= c.α + c.β[1] * x_next[1])
        end
    end

    # Objective: minimize -ω * x + θ  (profit - future value)
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


# ---------- Time-0 builder (deterministic) with ROI = c0 ----------
function build_stage0_markov(_t::Int,
                             vf_next::ValueFn{Float64},
                             _ω=nothing;
                             fix_state::Vector{Float64},
                             c0::Float64,
                             ucap0::Float64 = Inf)

    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, x_state[1:1])
    @variable(model, 0 <= u0)          # here-and-now decision
    @variable(model, x_next[1:1] >= 0)
    @variable(model, θ)

    @constraint(model, x_state .== fix_state)
    @constraint(model, u0 <= x_state[1])
    if isfinite(ucap0)
        @constraint(model, u0 <= ucap0)
    end
    eq = @constraint(model, x_next[1] == x_state[1] - u0)

    # Epigraph of V_{t+1}^{ctx} (continuation value fn passed in as vf_next)
    if isempty(vf_next.cuts)
        @constraint(model, θ >= 0)
    else
        for c in vf_next.cuts
            @constraint(model, θ >= c.α + c.β[1] * x_next[1])
        end
    end

    # Deterministic ROI at stage 0: c0 = 1.06 here
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



struct WeatherShock
    roi::Float64
    z_next::Int
end

# Deterministic stage: ctx is carried through unchanged, no ω randomness
Stage_deterministic_markov(; t, state_dim, build, node_key = z->z) = Stage(
    t,
    state_dim,
    build,
    # sampler(ctx) -> a dummy shock (nothing)
    ctx -> nothing,
    # weight(ω) (unused)
    _ -> 1.0,
    # children(t, ctx) -> single dummy shock with prob 1
    (tt, ctx) -> (Any[nothing], [1.0]),
    # next_ctx(t, ctx, ω) = ctx (no Markov transition at deterministic stage)
    (tt, ctx, _ω) -> ctx,
    # node_key(ctx)
    node_key,
)


# Build a Markov-aware Stage for the ice-cream problem
function make_markov_stage(t::Int; ucap::Float64)
    # builder: takes WeatherShock and passes roi into the stage LP builder
    build = (tt, vf_next, ω::WeatherShock; fix_state=nothing) ->
        build_stage_model(tt, vf_next, ω.roi; fix_state, ucap=ucap)

    # sampler: sample WeatherShock given current Markov state z
    sampler = (z::Int)->begin
        shocks, probs = Xi_of(z)
        r = rand()
        c = 0.0
        for (ω, p) in zip(shocks, probs)
            c += p
            if r <= c
                return ω
            end
        end
        return shocks[end]  # fallback so we always return something
    end

    # weight: not used directly in Markov backward pass, but kept for API completeness
    weight = _->1.0

    # children: full enumeration of WeatherShock for given z, used in backward pass
    children = (tt, z)->Xi_of(z)  # returns (Vector{WeatherShock}, Vector{Float64})

    # next_ctx: Markov state transition extracted from the shock
    next_ctx = (tt, z, ω::WeatherShock)->ω.z_next

    # node_key: group nodes by Markov state
    node_key = z->z

    Stage(t, 1, build, sampler, weight, children, next_ctx, node_key)
end


c0    = 1.06          # deterministic ROI at stage 0
ucap0 = ucap + 0.0    # same cap as later, or choose something else

# Stage 0: deterministic, Markov state ctx is carried through (e.g. CLOUDY)
stage0 = Stage_deterministic_markov(
    t = 1,
    state_dim = 1,
    build = (tt, vf_next, ω; fix_state=nothing) -> begin
        fix_state === nothing && error("stage 0 needs fix_state = B0")
        build_stage0_markov(tt, vf_next, ω;
                            fix_state = fix_state,
                            c0       = c0,
                            ucap0    = ucap0)
    end,
    # node_key groups all visits at stage 0 with same ctx. We have ctx=CLOUDY,
    # so node_key(z) = z is fine.
    node_key = z -> z,
)

stages_markov = [make_markov_stage(t; ucap=ucap) for t in 1:T]

# Prepend stage 0 to your random Markov stages
stages0_T = vcat([stage0], stages_markov)

# IMPORTANT: build MarkovSDDP with stages0_T, not stages_markov
mM = MarkovSDDP(stages0_T; discount = γ)


using Random
Random.seed!(1234)

z0   = CLOUDY   # start in Markov state CLOUDY

res = run_markov_sddp!(mM;
    x0 = B0,
    ctx0 = z0,
    S = 1,
    max_iter = 1000,
    patience = 20,
    value_tol = 0.0,
    evaluate_stage = 1,   # stage 0
    evaluate_ctx   = z0,  # CLOUDY
)
println("Stopped after $(res.iters) iterations. Cuts per stage = ", res.cuts_per_stage)

# Continuation value function at stage 1 (Markov stage), ctx = CLOUDY
vf1_cloudy = get_V!(mM, 2, CLOUDY)  # stage index 2 because of prepended stage0

# Rebuild and solve the deterministic stage-0 problem
model0, x_state0, θ0, misc0 = build_stage0_markov(
    1,            # t index for stage0 (consistent with Stage definition)
    vf1_cloudy,
    nothing;
    fix_state = B0,
    c0       = c0,
    ucap0    = ucap0,
)
optimize!(model0)

x0_star = value(misc0[:u])
B1_star = value.(misc0[:x_next])
obj0    = objective_value(model0)

println("x0* = $x0_star,  B1* = $(B1_star[1]),  obj0 = $obj0")

