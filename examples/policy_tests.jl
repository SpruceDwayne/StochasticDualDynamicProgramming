using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
Pkg.instantiate()
Pkg.precompile()

using SDDPBAPE
using JuMP, HiGHS, Random
using CSV, DataFrames, Statistics
using StatsBase  # if you want random sampling
Random.seed!(1234)


# ============================================================
# Load ξ_t from CSV
#   Columns: regime; a_t_tilde; r_t_tilde; pi_year
# ============================================================

xi_path = joinpath(@__DIR__, "xi_file.csv")
df = CSV.read(xi_path, DataFrame; normalizenames=true)
rename!(df, Symbol.(names(df)))


# First get the regimes from the full file
reg_vals = sort(unique(df.regime))        # e.g. [0, 1]
const K = length(reg_vals)
reg_index = Dict(r => i for (i, r) in enumerate(reg_vals))

# Scenario reduction (optional)
max_scen = 20
const T_rand = 10

df_small = DataFrame()
for r in reg_vals
    rows = df[df.regime .== r, :]
    n = nrow(rows)
    idx = n > max_scen ? sample(1:n, max_scen; replace=false) : 1:n
    df_small = vcat(df_small, rows[idx, :])
end
df = df_small  # overwrite with reduced dataset


# ============================================================
# Regimes and Markov transition matrix
# ============================================================

reg_vals = sort(unique(df.regime))        # e.g. [0, 1]
const K = length(reg_vals)
reg_index = Dict(r => i for (i, r) in enumerate(reg_vals))

# 2×2 transition matrix P(z_next | z)
const Pz = [
            0.74  0.26;
            0.29  0.71;
            ]
@assert size(Pz) == (K, K)

# ============================================================
# Shock type and Xi_of(z)
# ============================================================

struct XiShock
    a::Float64
    r::Float64
    pi::Float64
    z_next::Int
end

function Xi_of(z::Int)
    # Which original regime label corresponds to Markov index z?
    reg_label = reg_vals[z]                    # e.g. 0 or 1
    rows = df[df.regime .== reg_label, :]
    N = nrow(rows)
    @assert N > 0 "No rows in CSV for regime $(reg_label)"

    shocks = XiShock[]
    probs  = Float64[]

    base_p = 1.0 / N   # empirical P(ξ | z)

    # For each possible next Markov state z_next
    for z_next in 1:K
        Pzz = Pz[z, z_next]        # P(z_next | z)
        Pzz == 0.0 && continue

        # For each (a,r,pi) row conditional on current regime z
        for i in 1:N
            a  = rows.a_t_tilde[i]
            rt = rows.r_t_tilde[i]
            π  = rows.pi_year[i]

            push!(shocks, XiShock(a, rt, π, z_next))
            push!(probs, Pzz * base_p)   # joint P(ξ, z_next | z)
        end
    end

    @assert abs(sum(probs) - 1.0) < 1e-10
    return shocks, probs
end

# ============================================================
# Model parameters
# ============================================================

const γ         = 1.0       # discount
const state_dim = 5         # [w_t, l_t, c_t, ltilde_t, x_{t-1}]

# Risk-neutral: weights on tracking inflation and funding
const α = 1.0    # weight on inflation tracking slack s_t^{inf}
const β = 1.5    # weight on funding slack s_t^{fund}
const λ_fund = 1.0  # funding level scaling in l_t a_t - λ w_t

# Initial conditions: L0=1, C0=1.1 L0, W0 = 1.1 * a0 * L0
L0 = 1.0
C0 = 0.99 * L0
a0 = mean(df.a_t_tilde)
W0 = 1.05 * a0 * L0

xprev0 = 0.0
x0_state = [W0, L0, C0, L0, xprev0]  # [w0, l0, c0, ltilde0, x_{-1}=0]

# ============================================================
# Stage builders
# ============================================================
kappa=0.7
kappa_inf = 0.98
"""
Stage t ≥ 1 (random).

State x_state = (w_{t-1}, l_{t-1}, c_{t-1}, ltilde_{t-1}, x_{t-2})

Shock ω = (a_t, r_t, π_t).

Dynamics:
  c_t      = c_{t-1} (1 + π_t)
  ltilde_t = ltilde_{t-1} (1 + π_t)
  l_t      = l_{t-1} + b_t
  w_t      = w_{t-1} + c_t - l_t + x_{t-1} r_t
  x_{t-1}  (state component 5) carried in x_state[5]
  new exposure x_t becomes x_{t} in next state: x_next[5] = x_t

Slacks:
  s_inf  ≥ ltilde_t - l_t
  s_fund ≥ l_t a_t - λ_fund w_t

Immediate cost: α s_inf + β s_fund + continuation θ.
"""
function build_stage_model(t::Int,
                           vf_next::ValueFn{Float64},
                           ω::XiShock;
                           fix_state::Vector{Float64})

    model = Model(HiGHS.Optimizer); set_silent(model)

    # State variables at t-1
    @variable(model, x_state[1:state_dim])

    # Next-state variables at t
    @variable(model, x_next[1:state_dim])

    # Decision variables
    @variable(model, b >= 0)            # bonus
    @variable(model, x >= 0)            # new exposure x_t
    @variable(model, s_inf  >= 0)
    @variable(model, s_fund >= 0)
    @variable(model, θ)

    @constraint(model, x_state .== fix_state)

    # Shorthand
    w_prev      = x_state[1]
    l_prev      = x_state[2]
    c_prev      = x_state[3]
    ltilde_prev = x_state[4]
    x_prev      = x_state[5]

    w_t      = x_next[1]
    l_t      = x_next[2]
    c_t      = x_next[3]
    ltilde_t = x_next[4]
    x_carry  = x_next[5]   # stored x_t

    # Dynamics
    c_eq      = @constraint(model, c_t      == c_prev      * (1.0 +kappa* ω.pi))
    ltilde_eq = @constraint(model, ltilde_t == ltilde_prev * (1.0 + ω.pi))
    l_eq      = @constraint(model, l_t      == l_prev      + b)
    w_eq      = @constraint(model, w_t      == w_prev + c_t - l_t + x_prev * ω.r)
    xcarry_eq = @constraint(model, x_carry  == x)

    # Exposure limit: x_t ≤ 1.5 * w_t
    @constraint(model, x <= 1.5 * w_t)

    # Slacks
    @constraint(model, s_inf  >= ltilde_t -kappa_inf* l_t)
    @constraint(model, s_fund >= l_t * ω.a - λ_fund * w_t)

    # Continuation value: θ ≥ V_{t+1}(x_next)
    if isempty(vf_next.cuts)
        @constraint(model, θ >= 0)
    else
        for c in vf_next.cuts
            @constraint(model, θ >= c.α + sum(c.β[j] * x_next[j] for j in 1:state_dim))
        end
    end

    # Cost: α s_inf + β s_fund + θ
    @objective(model, Min, α * s_inf + β * s_fund + θ)

    misc = Dict{Symbol,Any}(
    :x_state  => x_state,
    :x_next   => x_next,
    :xnext_eq => [w_eq, l_eq, c_eq, ltilde_eq, xcarry_eq],
    :c_x      => zeros(state_dim),
    :x_dec    => x,        # decision x_t
    :b        => b,        # decision b_t
    :s_inf    => s_inf,
    :s_fund   => s_fund,
)


    return model, x_state, θ, misc
end


"""
Deterministic stage 0.

Here we only pick the initial exposure x_0, everything else is fixed:
  (w0, l0, c0, ltilde0, x_{-1}=0) given in fix_state.

We propagate:
  x_next[1:4] = x_state[1:4]
  x_next[5]   = x_0

No immediate cost at t=0; we only see θ >= V_1(x_next).
"""
function build_stage0(t::Int,
                      vf_next::ValueFn{Float64},
                      _ω=nothing;
                      fix_state::Vector{Float64})

    model = Model(HiGHS.Optimizer); set_silent(model)

    @variable(model, x_state[1:state_dim])
    @variable(model, x_next[1:state_dim])

    @variable(model, x >= 0)  # initial exposure x_0
    @variable(model, θ)

    @constraint(model, x_state .== fix_state)

    # Shorthand
    w0      = x_state[1]
    l0      = x_state[2]
    c0      = x_state[3]
    ltilde0 = x_state[4]
    x_prev0 = x_state[5]

    # Dynamics at t=0: no return yet, only carry x_0
    w_eq0      = @constraint(model, x_next[1] == w0)
    l_eq0      = @constraint(model, x_next[2] == l0)
    c_eq0      = @constraint(model, x_next[3] == c0)
    ltilde_eq0 = @constraint(model, x_next[4] == ltilde0)
    xcarry_eq0 = @constraint(model, x_next[5] == x)

    # Optional bound from your text: x_0 ≤ W_0 - a_0
    #@constraint(model, x <= w0 - a0)
    # And also respect leverage bound same as later: x_0 ≤ 1.5 * w0
    @constraint(model, x <= 1.5 * w0)

    # Continuation value from stage 1
    if isempty(vf_next.cuts)
        @constraint(model, θ >= 0)
    else
        for c in vf_next.cuts
            @constraint(model, θ >= c.α + sum(c.β[j] * x_next[j] for j in 1:state_dim))
        end
    end

    @objective(model, Min, θ)

    misc = Dict{Symbol,Any}(
        :x_state  => x_state,
        :x_next   => x_next,
        :xnext_eq => [w_eq0, l_eq0, c_eq0, ltilde_eq0, xcarry_eq0],
        :c_x      => zeros(state_dim),
        :x_dec    => x,    # <-- initial exposure x_0 for "logging"
    )

    return model, x_state, θ, misc
end


# Deterministic Stage wrapper (for stage 0)
Stage_deterministic(; t, state_dim, build) = Stage(
    t,
    state_dim,
    build,
    ctx -> nothing,          # sampler (no randomness)
    _   -> 1.0,              # weight
    (tt, ctx) -> (Any[nothing], [1.0]),  # children
    (tt, ctx, _ω) -> ctx,    # next_ctx = ctx
    ctx -> ctx               # node_key
)


# Build Markov random Stage for t ≥ 1
function make_markov_stage(t::Int)
    build = (tt, vf_next, ω::XiShock; fix_state=nothing) ->
        build_stage_model(tt, vf_next, ω; fix_state=fix_state)

    sampler = z -> begin
        shocks, probs = Xi_of(z)
        r = rand()
        c = 0.0
        for (ω, p) in zip(shocks, probs)
            c += p
            if r <= c
                return ω
            end
        end
        return shocks[end]
    end

    weight   = _ -> 1.0
    children = (tt, z) -> Xi_of(z)
    next_ctx = (tt, z, ω::XiShock) -> ω.z_next

    return Stage(t, state_dim, build, sampler, weight, children, next_ctx, z -> z)
end

# ============================================================
# Assemble stages and build MarkovSDDP model
# ============================================================

stage0 = Stage_deterministic(
    t = 1, state_dim = state_dim,
    build = (tt, vf_next, ω; fix_state=nothing) ->
        build_stage0(tt, vf_next, ω; fix_state=fix_state)
)

stages_rand = [make_markov_stage(t) for t in 2:(T_rand+1)]
stages = vcat([stage0], stages_rand)

m = MarkovSDDP(stages; discount = γ)

# ============================================================
# Run risk-neutral Markov SDDP
# ============================================================


# pick initial regime (use the first one in reg_vals)
z0_reg = reg_vals[1]
z0     = reg_index[z0_reg]



####Then to examine the policy we simulate it and store it as a CSV to be analysed elsewhere
using DataFrames,CSV

function simulate_policy(m::MarkovSDDP;
                         x0::Vector{Float64},
                         ctx0,
                         T::Int = m.T,
                         Nsim::Int = 100,
                         rng = Random.default_rng())

    rows = DataFrame(
        scenario = Int[],
        t        = Int[],
        regime   = Int[],        # current Markov state z_t (before transition)
        w_prev   = Float64[],
        l_prev   = Float64[],
        c_prev   = Float64[],
        ltilde_prev = Float64[],
        x_prev   = Float64[],
        # decisions taken at stage t
        x_dec    = Float64[],
        b_dec    = Union{Missing,Float64}[],
        # post-decision / post-dynamics state
        w        = Float64[],
        l        = Float64[],
        c        = Float64[],
        ltilde   = Float64[],
        x_carry  = Float64[],    # equals x_dec
        # shocks
        a        = Union{Missing,Float64}[],
        r        = Union{Missing,Float64}[],
        pi       = Union{Missing,Float64}[],
        # slacks
        s_inf    = Union{Missing,Float64}[],
        s_fund   = Union{Missing,Float64}[],
        # diagnostics
        FR       = Union{Missing,Float64}[],  # funding ratio at t: w_t/(a_t*l_t)
        IR       = Union{Missing,Float64}[],  # indexation ratio at t: l_t/ltilde_t
    )

    for s in 1:Nsim
        x_state = copy(x0)   # [w, l, c, ltilde, x_prev]
        ctx     = ctx0       # Markov state

        for t in 1:T
            stage = m.stages[t]
            ω = stage.sampler(ctx)  # `nothing` at stage 1 (deterministic)

            vf_next = if t < m.T
                get_V!(m, t+1, ctx)
            else
                ValueFn{Float64}()
            end

            model, _, _, misc = stage.build(t, vf_next, ω; fix_state = x_state)
            optimize!(model)

            x_next = value.(misc[:x_next])  # [w_t, l_t, c_t, ltilde_t, x_t]

            # pre
            w_prev, l_prev, c_prev, ltilde_prev, x_prev = x_state

            # decisions
            x_dec = x_next[5]
            b_val = haskey(misc, :b) ? value(misc[:b]) : missing

            s_inf  = haskey(misc, :s_inf)  ? value(misc[:s_inf])  : missing
            s_fund = haskey(misc, :s_fund) ? value(misc[:s_fund]) : missing

            a_val  = (ω isa XiShock) ? ω.a  : missing
            r_val  = (ω isa XiShock) ? ω.r  : missing
            pi_val = (ω isa XiShock) ? ω.pi : missing

            # post
            w_t, l_t, c_t, ltilde_t, x_carry = x_next

            # diagnostics (only defined when we have a_t)
            FR = (ω isa XiShock) ? (w_t / (ω.a * l_t)) : missing
            IR = l_t / ltilde_t

            push!(rows, (
                s, t, ctx,
                w_prev, l_prev, c_prev, ltilde_prev, x_prev,
                x_dec, b_val,
                w_t, l_t, c_t, ltilde_t, x_carry,
                a_val, r_val, pi_val,
                s_inf, s_fund,
                FR, IR
            ))

            x_state = x_next
            ctx     = stage.next_ctx(t, ctx, ω)
        end
    end

    return rows
end

function build_markov_model(; T_rand::Int, state_dim::Int=5, γ::Float64=1.0)
    stage0 = Stage_deterministic(
        t = 1, state_dim = state_dim,
        build = (tt, vf_next, ω; fix_state=nothing) ->
            build_stage0(tt, vf_next, ω; fix_state=fix_state)
    )
    stages_rand = [make_markov_stage(t) for t in 2:(T_rand+1)]
    stages = vcat([stage0], stages_rand)
    return MarkovSDDP(stages; discount = γ)
end

function risk_neutral_expected_cost(df_sim::DataFrame; α::Float64, β::Float64, γ::Float64=1.0)
    # stage cost only defined where slacks exist (random stages)
    # treat missing slacks as 0 just in case
    s_inf  = coalesce.(df_sim.s_inf,  0.0)
    s_fund = coalesce.(df_sim.s_fund, 0.0)

    stage_cost = α .* s_inf .+ β .* s_fund

    # discount by stage index t (your stage 1 is deterministic)
    disc = γ .^ (df_sim.t .- 1)

    total_cost_by_row = disc .* stage_cost

    # Average over scenarios of (sum over t)
    # Each scenario has multiple rows: group then sum.
    g = groupby(DataFrame(cost=total_cost_by_row, scenario=df_sim.scenario), :scenario)
    scenario_costs = combine(g, :cost => sum => :J).J

    return mean(scenario_costs)
end




alpha_risk = 0.80
lambdas = [0.0,0.1,0.2, 0.3,0.4, 0.5,0.6,0.7, 0.8,0.9,1.0]   # pick your grid

# initial regime
z0_reg = reg_vals[1]
z0     = reg_index[z0_reg]

# run settings (keep smaller to be fast)
max_iter = 1500
patience = 100

summary = DataFrame(
    lambda_risk = Float64[],
    iters = Int[],
    cuts_total = Int[],
    P_FR_T_lt_1 = Float64[],
    P_IR_T_lt_1 = Float64[],
    E_FR_T = Float64[],
    E_IR_T = Float64[],
    E_cost_RN = Float64[],    # <-- NEW: risk-neutral expected cost
)


for λρ in lambdas
    println("\n=== Solving for lambda_risk = $λρ ===")

    m = build_markov_model(T_rand=T_rand, state_dim=state_dim, γ=γ)

    res = run_markov_sddp_rho!(m;
        x0             = x0_state,
        ctx0           = z0,
        S              = 1,
        max_iter       = max_iter,
        patience       = patience,
        value_tol      = 0.005,
        evaluate_stage = 1,
        evaluate_ctx   = z0,
        alpha          = alpha_risk,
        lambda         = λρ
    )

    # include deterministic stage 1 + T_rand random stages:
    sim_horizon = T_rand + 1
    Random.seed!(2025)

    df_sim = simulate_policy(m; x0=x0_state, ctx0=z0, T=sim_horizon, Nsim=1000)

    df_T = df_sim[df_sim.t .== sim_horizon, :]
    # FR is missing at t=1 but should be defined at terminal (random stage), so safe here
    FR_T = skipmissing(df_T.FR)
    IR_T = skipmissing(df_T.IR)

    p_FR = mean(collect(FR_T) .< 1.0)
    p_IR = mean(collect(IR_T) .< 1.0)

    E_FR = mean(collect(FR_T))
    E_IR = mean(collect(IR_T))
    E_cost_RN = risk_neutral_expected_cost(df_sim; α=α, β=β, γ=γ)

    cuts_total = sum(res.cuts_per_stage)

    push!(summary, (
    λρ, res.iters, cuts_total,
    p_FR, p_IR,
    E_FR, E_IR,
    E_cost_RN
))


    # write simulation csv for plotting later
    out_csv = "sim_paths_T10_Xi20_lambda$(round(λρ, digits=2))_alpha$(alpha_risk).csv"
    CSV.write(out_csv, df_sim)

    println("iters=$(res.iters), P(FR_T<1)=$(round(p_FR, digits=4)), P(IR_T<1)=$(round(p_IR, digits=4))")
end

CSV.write("risk_aversion_sweep_summary_alpha08.csv", summary)
println("\nWrote risk_aversion_sweep_summary.csv")

 