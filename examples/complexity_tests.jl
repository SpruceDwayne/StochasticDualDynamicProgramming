###############################
# Complexity parameter sweep script
###############################
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
Pkg.instantiate()
Pkg.precompile()

using SDDPBAPE
using JuMP, HiGHS, Random
using CSV, DataFrames, Statistics
using StatsBase

Random.seed!(1234)

# ============================================================
# Load ξ_t from CSV
#   Expected columns (after normalizenames=true):
#     :regime, :a_t_tilde, :r_t_tilde, :pi_year
# ============================================================

xi_path = joinpath(@__DIR__, "xi_file.csv")
df = CSV.read(xi_path, DataFrame; normalizenames=true)
rename!(df, Symbol.(names(df)))

# Keep a full copy; we will rebuild a reduced df per run
df_full = df

# Identify regimes and map to Markov indices 1..K
reg_vals = sort(unique(df_full.regime))
const K = length(reg_vals)
const reg_index = Dict(r => i for (i, r) in enumerate(reg_vals))

# ============================================================
# Markov transition matrix P(z_next | z)
# ============================================================

const Pz = [
    0.74  0.26;
    0.29  0.71;
]
@assert size(Pz) == (K, K)

# ============================================================
# Scenario reduction helper (controls |Xi| per regime)
# ============================================================

function reduce_df_by_regime_old(df_full::DataFrame, reg_vals, max_scen::Int)
    df_small = DataFrame()
    for r in reg_vals
        rows = df_full[df_full.regime .== r, :]
        n = nrow(rows)
        idx = n > max_scen ? sample(1:n, max_scen; replace=false) : 1:n
        df_small = vcat(df_small, rows[idx, :])
    end
    return df_small
end

function reduce_df_by_regime(
    df_full::DataFrame,
    reg_vals,
    max_scen::Int;
    rng::AbstractRNG = Random.default_rng()
)
    df_small = DataFrame()
    for r in reg_vals
        rows = df_full[df_full.regime .== r, :]
        n = nrow(rows)
        idx = n > max_scen ? sample(rng, 1:n, max_scen; replace=false) : 1:n
        df_small = vcat(df_small, rows[idx, :])
    end
    return df_small
end

# ============================================================
# Shock type and Xi_of(z)
#   NOTE: Xi_of closes over global `df` (which we overwrite per run)
# ============================================================

struct XiShock
    a::Float64
    r::Float64
    pi::Float64
    z_next::Int
end

function Xi_of(z::Int)
    # global df must exist
    reg_label = reg_vals[z]
    rows = df[df.regime .== reg_label, :]
    N = nrow(rows)
    @assert N > 0 "No rows in CSV for regime $(reg_label)"

    shocks = XiShock[]
    probs  = Float64[]

    base_p = 1.0 / N  # empirical P(ξ | z)

    for z_next in 1:K
        Pzz = Pz[z, z_next]
        Pzz == 0.0 && continue

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

children_count = [length(first(Xi_of(z))) for z in 1:K]

# ============================================================
# Model parameters (kept fixed across complexity runs)
# ============================================================

const γ         = 1.0
const state_dim = 5  # [w, l, c, ltilde, x_prev]

const α = 1.0
const β = 1.5
const λ_fund = 1.0

kappa = 0.7
kappa_inf = 0.98

# Initial conditions (keep fixed for clean complexity comparisons)
L0 = 1.0
C0 = 0.99 * L0
a0_full = mean(df_full.a_t_tilde)
W0 = 0.999* a0_full * L0 #1.05 * a0_full * L0

xprev0 = 0.0
x0_state = [W0, L0, C0, L0, xprev0]

# ============================================================
# Stage builders
# ============================================================

function build_stage_model(t::Int,
                           vf_next::ValueFn{Float64},
                           ω::XiShock;
                           fix_state::AbstractVector{<:Real})

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x_state[1:state_dim])
    @variable(model, x_next[1:state_dim])

    @variable(model, b >= 0)
    @variable(model, x >= 0)
    @variable(model, s_inf  >= 0)
    @variable(model, s_fund >= 0)
    @variable(model, θ)

    @constraint(model, x_state .== fix_state)

    w_prev      = x_state[1]
    l_prev      = x_state[2]
    c_prev      = x_state[3]
    ltilde_prev = x_state[4]
    x_prev      = x_state[5]

    w_t      = x_next[1]
    l_t      = x_next[2]
    c_t      = x_next[3]
    ltilde_t = x_next[4]
    x_carry  = x_next[5]

    c_eq      = @constraint(model, c_t      == c_prev      * (1.0 + kappa * ω.pi))
    ltilde_eq = @constraint(model, ltilde_t == ltilde_prev * (1.0 + ω.pi))
    l_eq      = @constraint(model, l_t      == l_prev      + b)
    w_eq      = @constraint(model, w_t      == w_prev + c_t - l_t + x_prev * ω.r)
    xcarry_eq = @constraint(model, x_carry  == x)

    @constraint(model, x <= 1.5 * w_t)

    @constraint(model, s_inf  >= ltilde_t - kappa_inf * l_t)
    @constraint(model, s_fund >= l_t * ω.a - λ_fund * w_t)

    if isempty(vf_next.cuts)
        @constraint(model, θ >= 0)
    else
        for c in vf_next.cuts
            @constraint(model, θ >= c.α + sum(c.β[j] * x_next[j] for j in 1:state_dim))
        end
    end

    @objective(model, Min, α * s_inf + β * s_fund + θ)

    misc = Dict{Symbol,Any}(
        :x_state => x_state,
        :x_next  => x_next,
        :xnext_eq => [w_eq, l_eq, c_eq, ltilde_eq, xcarry_eq],
        :x_dec   => x,
        :b       => b,
        :s_inf   => s_inf,
        :s_fund  => s_fund,
    )

    return model, x_state, θ, misc
end

function build_stage0(t::Int,
                      vf_next::ValueFn{Float64},
                      _ω=nothing;
                      fix_state::AbstractVector{<:Real})

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x_state[1:state_dim])
    @variable(model, x_next[1:state_dim])
    @variable(model, x >= 0)
    @variable(model, θ)

    @constraint(model, x_state .== fix_state)

    w0      = x_state[1]
    l0      = x_state[2]
    c0      = x_state[3]
    ltilde0 = x_state[4]

    w_eq0      = @constraint(model, x_next[1] == w0)
    l_eq0      = @constraint(model, x_next[2] == l0)
    c_eq0      = @constraint(model, x_next[3] == c0)
    ltilde_eq0 = @constraint(model, x_next[4] == ltilde0)
    xcarry_eq0 = @constraint(model, x_next[5] == x)

    @constraint(model, x <= 1.5 * w0)

    if isempty(vf_next.cuts)
        @constraint(model, θ >= 0)
    else
        for c in vf_next.cuts
            @constraint(model, θ >= c.α + sum(c.β[j] * x_next[j] for j in 1:state_dim))
        end
    end

    @objective(model, Min, θ)

    misc = Dict{Symbol,Any}(
        :x_state => x_state,
        :x_next  => x_next,
        :xnext_eq => [w_eq0, l_eq0, c_eq0, ltilde_eq0, xcarry_eq0],
        :x_dec   => x,
    )

    return model, x_state, θ, misc
end

Stage_deterministic(; t, state_dim, build) = Stage(
    t,
    state_dim,
    build,
    ctx -> nothing,
    _   -> 1.0,
    (tt, ctx) -> (Any[nothing], [1.0]),
    (tt, ctx, _ω) -> ctx,
    ctx -> ctx
)

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

function build_markov_model(; T_rand::Int, state_dim::Int=5, γ::Float64=1.0)
    stage0 = Stage_deterministic(
        t = 1, state_dim = state_dim,
        build = (tt, vf_next, ω; fix_state=nothing) ->
            build_stage0(tt, vf_next, ω; fix_state=fix_state)
    )
    stages_rand = [make_markov_stage(t) for t in 2:(T_rand+1)]
    stages = vcat([stage0], stages_rand)
    return MarkovSDDP(stages; discount=γ)
end

# ============================================================
# Utilities: robust extraction from NamedTuples in res.history
# ============================================================

getprop(nt, sym::Symbol, default) = hasproperty(nt, sym) ? getproperty(nt, sym) : default

function history_to_df(hist)
    DataFrame(
        iter = [getprop(h, :iter, missing) for h in hist],
        new_cuts = [getprop(h, :new_cuts, missing) for h in hist],
        total_cuts = [getprop(h, :total_cuts, missing) for h in hist],
        V = [getprop(h, :V, missing) for h in hist],
        ΔV = [getprop(h, :ΔV, missing) for h in hist],

        t_fwd = [getprop(h, :t_fwd, missing) for h in hist],
        t_bwd = [getprop(h, :t_bwd, missing) for h in hist],
        t_eval = [getprop(h, :t_eval, missing) for h in hist],
        t_iter = [getprop(h, :t_iter, missing) for h in hist],
        t_cum  = [getprop(h, :t_cum, missing) for h in hist],

        unique_nodes = [getprop(h, :unique_nodes, missing) for h in hist],
        bwd_work_units = [getprop(h, :bwd_work_units, missing) for h in hist],
        avg_children_visited = [getprop(h, :avg_children_visited, missing) for h in hist],

        # per_stage is a vector; store it as a string so CSV writes cleanly
        per_stage = [string(getprop(h, :per_stage, missing)) for h in hist],
    )
end


function summarize_run(res; T_rand::Int, S::Int, max_scen::Int)
    hist = res.history
    hdf = history_to_df(hist)

    # totals
    t_total = sum(skipmissing(hdf.t_iter))
    t_fwd   = sum(skipmissing(hdf.t_fwd))
    t_bwd   = sum(skipmissing(hdf.t_bwd))
    t_eval  = sum(skipmissing(hdf.t_eval))

    # distribution of iteration time
    t_iters = collect(skipmissing(hdf.t_iter))
    t_iter_mean = isempty(t_iters) ? missing : mean(t_iters)
    t_iter_med  = isempty(t_iters) ? missing : median(t_iters)
    t_iter_p90  = isempty(t_iters) ? missing : quantile(t_iters, 0.90)

    cuts_total = (nrow(hdf) == 0) ? missing : hdf.total_cuts[end]
    V_final    = (nrow(hdf) == 0) ? missing : hdf.V[end]

    return (
        T_rand=T_rand,
        S=S,
        max_scen=max_scen,
        iters=res.iters,
        cuts_total=cuts_total,
        V_final=V_final,
        t_total=t_total,
        t_fwd=t_fwd,
        t_bwd=t_bwd,
        t_eval=t_eval,
        t_iter_mean=t_iter_mean,
        t_iter_med=t_iter_med,
        t_iter_p90=t_iter_p90
    )
end

# ============================================================
# Complexity sweep configuration
# ============================================================

# Fix risk parameters (keep constant during complexity study)
alpha_risk  = 0.90
lambda_risk = 0.5

# Training controls (keep constant across runs)
max_iter    = 1000
patience    = 200
value_tol   = 0.0025
force_every = 10
cut_atol    = 1e-8

# Initial regime: use the first label in reg_vals
z0_reg = reg_vals[2]
z0     = reg_index[z0_reg]

# Grids
T_grid  = []#[4,6, 8, 10, 12, 15]        # random years
S_grid  = [1,2, 5, 10, 20,40]         # forward trajectories per iteration
Xi_grid = []#[10, 20, 30, 40]      # empirical rows per regime (max_scen)

# Build an experiment list varying one dimension at a time
experiments = NamedTuple[]

# (1) vary T, keep S and Xi fixed
S_fix = 10
Xi_fix = 10
for Tt in T_grid
    push!(experiments, (T_rand=Tt, S=S_fix, max_scen=Xi_fix, tag="varyT"))
end

# (2) vary S, keep T and Xi fixed
T_fix = 8
Xi_fix2 = 20
for Ss in S_grid
    push!(experiments, (T_rand=T_fix, S=Ss, max_scen=Xi_fix2, tag="varyS"))
end

# (3) vary Xi, keep T and S fixed
T_fix2 = 2#8
S_fix2 = 1
for Xx in Xi_grid
    push!(experiments, (T_rand=T_fix2, S=S_fix2, max_scen=Xx, tag="varyXi"))
end

# Output directory
outdir = joinpath(@__DIR__, "results_complexity")
mkpath(outdir)

summary_rows = NamedTuple[]

# ============================================================
# Run sweep
# ============================================================

for (run_id, cfg) in enumerate(experiments)
    Tt = cfg.T_rand
    Ss = cfg.S
    Xx = cfg.max_scen
    tag = cfg.tag

    println("\n=== Run $run_id / $(length(experiments)) [$tag] | T=$Tt, S=$Ss, Xi_per_regime=$Xx ===")

    # Rebuild reduced dataset (controls |Xi| per regime)
    #global df = reduce_df_by_regime(df_full, reg_vals, Xx)
    global df = reduce_df_by_regime(df_full, reg_vals, Xx; rng=Random.seed!(1234))
    Random.seed!(1234)

    # Build model with T_rand stages (plus deterministic stage 0 = stage index 1)
    m = build_markov_model(T_rand=Tt, state_dim=state_dim, γ=γ)

    # Train with your timing-enabled run_markov_sddp_rho! (assumed already defined/loaded)
    K = 2
    res = run_markov_sddp_rho!(m;
        x0=x0_state,
        ctx0=z0,
        K=K,
        children_count=children_count,
        S=Ss,
        max_iter=max_iter,
        patience=10+div(200,Ss),
        value_tol=value_tol,
        evaluate_stage=1,
        evaluate_ctx=z0,
        force_every=force_every,
        cut_atol=cut_atol,
        alpha=alpha_risk,
        lambda=lambda_risk,
        VALUE_CHECK_WINDOW=10+div(200,Ss)
    )

    # Save per-iteration history
    hist_df = history_to_df(res.history)
    hist_path = joinpath(outdir, "hist_$(tag)_T$(Tt)_S$(Ss)_Xi$(Xx).csv")
    CSV.write(hist_path, hist_df)

    # Save one-line summary
    push!(summary_rows, merge(summarize_run(res; T_rand=Tt, S=Ss, max_scen=Xx), (tag=tag,)))

    # Print quick KPI
    t_total = sum(skipmissing(hist_df.t_iter))
    cuts_total = hist_df.total_cuts[end]
    println("iters=$(res.iters) | cuts=$(cuts_total) | total time=$(round(t_total, digits=2))s | wrote $(basename(hist_path))")
end

summary_df = DataFrame(summary_rows)
summary_path = joinpath(outdir, "complexity_summary.csv")
CSV.write(summary_path, summary_df)

println("\nDone.")
println("Wrote: $(summary_path)")
println("Wrote: $(length(experiments)) history CSV files in $(outdir)")
