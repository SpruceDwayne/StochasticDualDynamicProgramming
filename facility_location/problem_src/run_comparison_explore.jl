import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "SDDPBAPE"))
for pkg in ["CSV", "DataFrames", "Plots", "Statistics"]
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end

using SDDPBAPE
using JuMP, Gurobi

const GRB_ENV = Gurobi.Env(output_flag = 0)
using Printf, Statistics, Random
using CSV, DataFrames, Plots

include(joinpath(@__DIR__, "distributions.jl"))
include(joinpath(@__DIR__, "subproblem_G.jl"))

# ── CLI arguments ─────────────────────────────────────────────────────────────
# Usage: julia run_comparison_explore.jl [TYPE] [N_SAA]
#   TYPE  : instance type A / B / C / D  (default: D)
#   N_SAA : number of SAA scenarios      (default: 10)
const ITYPE = length(ARGS) >= 1 ? uppercase(strip(ARGS[1])) : "D"
@assert ITYPE in ("A","B","C","D") "Unknown interaction type '$ITYPE'. Use A, B, C or D."

# ── Shared setup ──────────────────────────────────────────────────────────────
const INST_PATH = joinpath(@__DIR__, "..", "..", "instance_data_$(ITYPE).json")
const inst = load_instance(INST_PATH)
@printf("Instance type: %s | %d facilities | %d customers | %d zones | T=%d\n",
    inst.interaction_type, inst.num_facilities, inst.num_customers, inst.num_zones, inst.T)

const N_SAA     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const saa       = generate_saa_scenarios(inst; N=N_SAA, seed=123)
const nI        = inst.num_facilities
const n_regions = length(inst.activation_regions)
const uniform_p = fill(1.0 / N_SAA, N_SAA)
const x0        = zeros(Float64, nI)
const ζ_init    = 1
const M_BIG     = 1e6

const ddu_regions = DDURegion[
    DDURegion(d, Any[saa[d][n, :] for n in 1:N_SAA], uniform_p)
    for d in 1:n_regions
]

const zone_active_0 = region_to_zone_active(inst, 1)
const pmfs_region1  = compute_demand_distributions(inst, zone_active_0)
const expected_d1   = Float64[
    sum(Float64(k) * pmfs_region1[j][k+1] for k in 0:inst.max_demand)
    for j in 1:inst.num_customers
]

const unit_weight = _ -> 1.0
const noop_ctx    = (t, ζ, ω) -> ζ
const noop_key    = x -> x

# ── Exogenous stage builder (no DDU variables, used for SDDiP baseline) ───────
function make_exogenous_stage_builder(inst::InstanceData, t::Int, grb_env::Gurobi.Env)
    nI_  = inst.num_facilities
    nJ_  = inst.num_customers
    P_   = inst.profit_matrix
    O_   = Float64(inst.O)
    C_   = Float64(inst.C)
    k_   = inst.k
    is_root_     = (t == 1)
    is_terminal_ = (t == inst.T)
    theta_lb_ = is_terminal_ ? 0.0 : -Float64(inst.T - t) * 50_000.0

    function builder(_t, vf_next, ω; fix_state)
        model  = Model(() -> Gurobi.Optimizer(grb_env))
        set_silent(model)

        @variable(model, x[1:nI_], Bin)

        z_refs = JuMP.VariableRef[]
        if !is_root_
            @variable(model, z[1:nI_], lower_bound = 0.0, upper_bound = 1.0)
            for i in 1:nI_
                @constraint(model, z[i] == fix_state[i])
            end
            append!(z_refs, z)

            for i in 1:nI_
                @constraint(model, x[i] >= z_refs[i])
            end
            @constraint(model, sum(x[i] - z_refs[i] for i in 1:nI_) <= k_)

            demand = Float64[ω[j] for j in 1:nJ_]
            @variable(model, w[1:nI_, 1:nJ_] >= 0)
            for j in 1:nJ_
                @constraint(model, sum(w[i, j] for i in 1:nI_) <= demand[j])
            end
            for i in 1:nI_
                @constraint(model, sum(w[i, j] for j in 1:nJ_) <= C_ * z_refs[i])
            end
        else
            @constraint(model, sum(x) <= k_)
        end

        if is_terminal_
            @variable(model, θ == 0.0)
        else
            @variable(model, θ >= theta_lb_)
            x_next_ref = JuMP.VariableRef[x[i] for i in 1:nI_]
            for c in vf_next.cuts
                @constraint(model, θ >= c.α + sum(c.β[j] * x_next_ref[j] for j in 1:nI_))
            end
        end

        if is_root_
            @objective(model, Min, O_ * sum(x) + θ)
        else
            opening = O_ * sum(x[i] - z_refs[i] for i in 1:nI_)
            @objective(model, Min,
                -sum(P_[i, j] * w[i, j] for i in 1:nI_, j in 1:nJ_) + opening + θ)
        end

        misc = Dict{Symbol, Any}(:x_next => JuMP.VariableRef[x[i] for i in 1:nI_])
        if !is_root_
            misc[:z_vars] = z_refs
        end

        return model, z_refs, θ, misc
    end

    return builder
end

function compute_std_lb!(m_std::SDDP, x0::Vector{Float64}, ω1, inst)
    vf2   = m_std.T >= 2 ? m_std.V[2] : ValueFn{Float64}()
    model1, _, _, _ = SDDPBAPE.get_or_build_model!(m_std, 1, vf2, ω1, nothing, x0)
    JuMP.optimize!(model1)
    return JuMP.objective_value(model1)
end

# ── Exploration infrastructure ─────────────────────────────────────────────────

"""
    build_exploration_map(inst) -> Vector{Tuple{Vector{Float64}, Int}}

Enumerate all binary facility vectors x with sum(x) ≤ k (maximum first-stage
openings).  Group by outgoing region (via `identify_region`) and keep one
representative per distinct reachable region.

Returns a vector of (x_representative, region_id) pairs.
"""
function build_exploration_map(inst::InstanceData)
    nI_ = inst.num_facilities
    k_  = inst.k   # max new openings at stage 1 (x0 = 0 so sum(x) <= k)

    region_to_x = Dict{Int, Vector{Float64}}()

    # Iterate over all 2^nI binary masks, filter by opening budget
    for mask in 0:(2^nI_ - 1)
        bits = [(mask >> (i-1)) & 1 for i in 1:nI_]
        sum(bits) > k_ && continue

        d = identify_region(inst, bits)
        if !haskey(region_to_x, d)
            region_to_x[d] = Float64.(bits)
        end
    end

    # Sort by region id for deterministic ordering
    pairs = sort(collect(region_to_x); by = first)
    return [(x, d) for (d, x) in pairs]
end

# Local copy of _active_config (unexported internal in SDDPBAPE)
_adapt_config(cfg::SDDiPConfig, iter::Int) =
    (cfg.burnin_iters > 0 && iter <= cfg.burnin_iters &&
     cfg.burnin_cut_type !== cfg.cut_type) ?
    SDDiPConfig(cfg.burnin_cut_type, cfg.lag_tol, cfg.lag_max_iter,
                cfg.step_size_init, cfg.step_decay, 0, cfg.burnin_cut_type, nothing) :
    cfg

# Local helper: argmax over region indicators (mirrors _read_active_region)
function _read_active_region_local(region_indicators::Dict{Int, JuMP.VariableRef})
    isempty(region_indicators) && return 0
    best_id  = first(keys(region_indicators))
    best_val = -Inf
    for (id, var) in region_indicators
        v = JuMP.value(var)
        if v > best_val
            best_val = v
            best_id  = id
        end
    end
    return best_id
end

"""
    forward_pass_ddu_forced!(m, x1_forced, d1_forced, x0, ζ_init) -> DDUForwardRecord

Run a single-path DDU forward pass where the stage-1 decision is forced to
`x1_forced` (region `d1_forced`) without solving the stage-1 subproblem.
Stages 2..T are solved normally.

This lets the backward pass generate cuts for region `d1_forced` even when the
stage-1 optimizer would not spontaneously choose that region.
"""
function forward_pass_ddu_forced!(
    m::DDUSDDP,
    x1_forced::Vector{Float64},
    d1_forced::Int,
    x0::Vector{Float64},
    ζ_init::Int,
)
    T = m.T
    S = 1  # single forced path

    x_entering = [[zeros(Float64, m.stages[t].state_dim) for t in 1:T] for _ in 1:S]
    x_leaving  = [[zeros(Float64, m.stages[t].state_dim) for t in 1:T] for _ in 1:S]
    δ_hist     = [[0 for _ in 1:T] for _ in 1:S]
    ζ_hist     = [[0 for _ in 1:T] for _ in 1:S]

    s = 1

    # Stage 1: force the decision without calling the optimizer
    x_entering[s][1] = copy(x0)
    ζ_hist[s][1]     = ζ_init
    x_leaving[s][1]  = copy(x1_forced)
    δ_hist[s][1]     = d1_forced

    x = x1_forced
    ζ = d1_forced

    # Stages 2..T: solve freely under the forced context
    for t in 2:T
        stg = m.stages[t]

        x_entering[s][t] = copy(x)
        ζ_hist[s][t]     = ζ

        ωt    = stg.sampler(ζ)
        model, _, _, misc = get_or_build_ddu_model!(m, t, ωt, x)
        JuMP.optimize!(model)

        status = JuMP.termination_status(model)
        status == MOI.OPTIMAL || @warn "forced forward pass: stage $t status = $status"

        x_out            = JuMP.value.(misc[:x_next])
        x_leaving[s][t]  = copy(x_out)

        ri = misc[:region_indicators] :: Dict{Int, JuMP.VariableRef}
        δt = _read_active_region_local(ri)
        δ_hist[s][t] = δt

        x = x_out
        ζ = δt
    end

    return DDUForwardRecord(x_entering, x_leaving, δ_hist, ζ_hist)
end

"""
    run_ddu_sddip_extra!(m; exploration_map, x0, ζ_init, config, S,
                          max_iter, patience, force_every, cut_atol,
                          explore_warmstart, explore_every, logfn)
        -> NamedTuple

DDU-SDDiP training loop with forced-exploration warm-start.

During the first `explore_warmstart` iterations AND every `explore_every`
iterations thereafter, a forced backward pass is run for every entry in
`exploration_map`.  Each forced pass uses a single path whose stage-1 decision
is overridden to the map's representative, ensuring cuts are generated for
every reachable region even if the online forward pass never visits it.

Set `explore_every = 0` to disable periodic re-exploration after warm-start.
"""
function run_ddu_sddip_extra!(
    m::DDUSDDP;
    exploration_map::Vector{Tuple{Vector{Float64}, Int}},
    x0::AbstractVector,
    ζ_init::Int,
    config::SDDiPConfig   = SDDiPConfig(),
    S::Int                = 1,
    max_iter::Int         = 500,
    patience::Int         = 100,
    force_every::Int      = 2,
    cut_atol::Float64     = 1e-8,
    explore_warmstart::Int = 20,
    explore_every::Int    = 0,
    logfn                 = nothing,
)
    total_cuts() = sum(
        t -> sum(vf -> length(vf.cuts), values(m.V[t]); init = 0),
        1:m.T; init = 0,
    )
    cuts_by_stage() = [
        sum(vf -> length(vf.cuts), values(m.V[t]); init = 0)
        for t in 1:m.T
    ]

    prev_total = total_cuts()
    stagnant   = 0
    x0f        = collect(Float64, x0)
    prev_lb    = compute_ddu_lb!(m, x0f, ζ_init)
    hist       = Vector{NamedTuple}()

    n_explore = length(exploration_map)

    for it in 1:max_iter
        cfg_it = _adapt_config(config, it)

        # ── Exploration phase ──────────────────────────────────────────────────
        do_explore = (it <= explore_warmstart) ||
                     (explore_every > 0 && ((it - explore_warmstart) % explore_every == 0))

        if do_explore
            for (x1f, d1f) in exploration_map
                fwd_forced = forward_pass_ddu_forced!(m, x1f, d1f, x0f, ζ_init)
                # force_every=0 so cuts are added only when genuinely improving
                backward_pass_ddu_sddip!(m;
                    fwd         = fwd_forced,
                    config      = cfg_it,
                    iter        = it,
                    force_every = 0,
                    atol        = cut_atol,
                )
            end
        end

        # ── Standard online forward-backward ──────────────────────────────────
        fwd = forward_pass_ddu_online!(m; S = S, x0 = x0f, ζ_init = ζ_init)
        backward_pass_ddu_sddip!(m;
            fwd         = fwd,
            config      = cfg_it,
            iter        = it,
            force_every = force_every,
            atol        = cut_atol,
        )

        cur_total = total_cuts()
        new_cuts  = cur_total - prev_total
        prev_total = cur_total

        cur_lb  = compute_ddu_lb!(m, x0f, ζ_init)
        Δlb     = cur_lb - prev_lb
        prev_lb = cur_lb

        stats = (iter = it, new_cuts = new_cuts, total_cuts = cur_total,
                 lb = cur_lb, Δlb = Δlb, per_stage = cuts_by_stage(),
                 explored = do_explore,
                 phase = it <= config.burnin_iters ? :burnin : :main)
        push!(hist, stats)

        exp_tag = do_explore ? "[exp] " : "       "
        if logfn === nothing
            @printf "iter %4d | %snew cuts: %2d | total: %3d | LB=%.6f | ΔLB=%.3e\n" it exp_tag new_cuts cur_total cur_lb Δlb
        else
            logfn(it, stats)
        end

        stagnant = (new_cuts == 0) ? (stagnant + 1) : 0
        if stagnant >= patience
            println("DDU-extra early stop: no new cuts for $patience consecutive iterations.")
            return (iters = it, cuts_per_stage = cuts_by_stage(), history = hist, lb = cur_lb)
        end
    end

    println("DDU-extra reached max_iter without early stop.")
    return (iters = max_iter, cuts_per_stage = cuts_by_stage(), history = hist, lb = prev_lb)
end

# ── Part 1: DDU-extra ─────────────────────────────────────────────────────────

stages_ddu = Stage[
    Stage(1, nI, make_stage_builder(inst, 1, GRB_ENV),
          (ζ) -> nothing,   unit_weight,
          (t, ζ) -> (ddu_regions[ζ].Xi, ddu_regions[ζ].pXi), noop_ctx, noop_key),
    Stage(2, nI, make_stage_builder(inst, 2, GRB_ENV),
          (ζ) -> ddu_regions[ζ].Xi[rand(1:N_SAA)], unit_weight,
          (t, ζ) -> (ddu_regions[ζ].Xi, ddu_regions[ζ].pXi), noop_ctx, noop_key),
    Stage(3, nI, make_stage_builder(inst, 3, GRB_ENV),
          (ζ) -> ddu_regions[ζ].Xi[rand(1:N_SAA)], unit_weight,
          (t, ζ) -> (ddu_regions[ζ].Xi, ddu_regions[ζ].pXi), noop_ctx, noop_key),
    Stage(4, nI, make_stage_builder(inst, 4, GRB_ENV),
          (ζ) -> ddu_regions[ζ].Xi[rand(1:N_SAA)], unit_weight,
          (t, ζ) -> (ddu_regions[ζ].Xi, ddu_regions[ζ].pXi), noop_ctx, noop_key),
]

regions_per_stage = Vector{Vector{DDURegion}}(undef, inst.T)
for t in 1:inst.T - 1
    regions_per_stage[t] = ddu_regions
end
regions_per_stage[inst.T] = DDURegion[]

m_ddu_extra = DDUSDDP(stages_ddu, regions_per_stage; M_big = M_BIG, discount = 1.0)

config = SDDiPConfig(
    cut_type        = :IO,
    burnin_iters    = 0,
    burnin_cut_type = :IO,
    level_cfg       = LevelMethodConfig(optimizer = () -> Gurobi.Optimizer(GRB_ENV)),
)

config2 = SDDiPConfig(
    cut_type        = :lagrangian,
    burnin_iters    = 0,
    burnin_cut_type = :IO,
    level_cfg       = LevelMethodConfig(optimizer = () -> Gurobi.Optimizer(GRB_ENV)),
)

# ── Build and print exploration map ───────────────────────────────────────────

println("\n" * "="^72)
println("Building exploration map ...")
println("="^72)

const exploration_map = build_exploration_map(inst)

println("Reachable regions at stage 1: $(length(exploration_map)) / $n_regions")
println("  (regions reachable with sum(x) ≤ $(inst.k) out of $(inst.num_facilities) facilities)")
println()
for (x1f, d1f) in exploration_map
    facs = findall(v -> v > 0.5, x1f)
    @printf("  Region %2d  ←  facilities %s\n", d1f, string(facs))
end

# ── Run DDU-extra training ─────────────────────────────────────────────────────

extra_lb_hist   = Float64[]
extra_time_hist = Float64[]
t_extra_start   = time()

function extra_logfn(it, stats)
    push!(extra_lb_hist,   stats.lb)
    push!(extra_time_hist, time() - t_extra_start)
    exp_tag = stats.explored ? "[exp] " : "       "
    @printf("DDU-extra iter %4d | %snew cuts: %2d | total: %3d | LB=%.4f | ΔLB=%.3e\n",
        it, exp_tag, stats.new_cuts, stats.total_cuts, stats.lb, stats.Δlb)
end

println("\n" * "="^72)
println("DDU-SDDiP + forced exploration (DDU-extra)")
println("="^72)

result_extra = run_ddu_sddip_extra!(m_ddu_extra;
    exploration_map   = exploration_map,
    x0                = x0,
    ζ_init            = ζ_init,
    config            = config,
    S                 = 1,
    max_iter          = 400,
    patience          = 100,
    force_every       = 2,
    cut_atol          = 1e-8,
    explore_warmstart = 20,
    explore_every     = 0,
    logfn             = extra_logfn,
)
t_extra = time() - t_extra_start

# Extract DDU-extra first-stage decision
ω1_extra = m_ddu_extra.stages[1].sampler(ζ_init)
model1_extra, _, _, misc1_extra = get_or_build_ddu_model!(m_ddu_extra, 1, ω1_extra, x0)
JuMP.optimize!(model1_extra)
x1_extra_vals = JuMP.value.(misc1_extra[:x_next])
x1_extra_int  = [v > 0.5 ? 1 : 0 for v in x1_extra_vals]
opened_extra  = findall(v -> v > 0.5, x1_extra_vals)
region1_extra = identify_region(inst, x1_extra_int)

@printf("\nDDU-extra done: %d iters | LB=%.4f | %.1f s\n",
    result_extra.iters, result_extra.lb, t_extra)
@printf("  Facilities opened: %s | region d=%d\n", string(opened_extra), region1_extra)

# ── Part 2: SDDiP (ignoring DDU) ─────────────────────────────────────────────

rng_std = MersenneTwister(456)

stages_std = Stage[
    Stage(1, nI, make_exogenous_stage_builder(inst, 1, GRB_ENV),
          ()  -> nothing, unit_weight,
          (t, ctx) -> (Any[saa[1][n, :] for n in 1:N_SAA], uniform_p),
          (t, ctx, ω) -> nothing, x -> nothing),
    Stage(2, nI, make_exogenous_stage_builder(inst, 2, GRB_ENV),
          ()  -> saa[1][rand(rng_std, 1:N_SAA), :], unit_weight,
          (t, ctx) -> (Any[saa[1][n, :] for n in 1:N_SAA], uniform_p),
          (t, ctx, ω) -> nothing, x -> nothing),
    Stage(3, nI, make_exogenous_stage_builder(inst, 3, GRB_ENV),
          ()  -> saa[1][rand(rng_std, 1:N_SAA), :], unit_weight,
          (t, ctx) -> (Any[saa[1][n, :] for n in 1:N_SAA], uniform_p),
          (t, ctx, ω) -> nothing, x -> nothing),
    Stage(4, nI, make_exogenous_stage_builder(inst, 4, GRB_ENV),
          ()  -> saa[1][rand(rng_std, 1:N_SAA), :], unit_weight,
          (t, ctx) -> (Any[saa[1][n, :] for n in 1:N_SAA], uniform_p),
          (t, ctx, ω) -> nothing, x -> nothing),
]

m_std = SDDP(stages_std)

std_lb_hist   = Float64[]
std_time_hist = Float64[]
t_std_start   = time()
ω1_std        = nothing

function std_logfn(it, stats)
    lb = compute_std_lb!(m_std, x0, ω1_std, inst)
    push!(std_lb_hist,   lb)
    push!(std_time_hist, time() - t_std_start)
    @printf("SDDiP iter %4d | new cuts: %2d | total: %3d | LB=%.4f | ΔV=%.3e\n",
        it, stats.new_cuts, stats.total_cuts, lb, stats.ΔV)
end

println("\n" * "="^72)
println("SDDiP (ignoring DDU — region 1 demand throughout)")
println("="^72)

result_std = run_sddip!(m_std;
    x0          = x0,
    ctx0        = nothing,
    config      = config2,
    S           = 1,
    max_iter    = 200,
    patience    = 20,
    force_every = 2,
    cut_atol    = 1e-6,
    logfn       = std_logfn,
)
t_std = time() - t_std_start

vf2_std      = m_std.V[2]
model1_std, _, _, misc1_std = SDDPBAPE.get_or_build_model!(m_std, 1, vf2_std, ω1_std, nothing, x0)
JuMP.optimize!(model1_std)
x1_std_vals  = JuMP.value.(misc1_std[:x_next])
x1_std_int   = [v > 0.5 ? 1 : 0 for v in x1_std_vals]
opened_std   = findall(v -> v > 0.5, x1_std_vals)
region1_std  = identify_region(inst, x1_std_int)
lb_std_final = JuMP.objective_value(model1_std)

@printf("\nSDDiP done: %d iters | LB=%.4f | %.1f s\n",
    result_std.iters, lb_std_final, t_std)
@printf("  Facilities opened: %s | region d=%d\n", string(opened_std), region1_std)

# ── Part 3: Out-of-sample simulation ─────────────────────────────────────────

const N_SIM = 1000

const _sim_saa_indices = let rng = MersenneTwister(20250101)
    [rand(rng, 1:N_SAA) for s in 1:N_SIM, t in 2:inst.T]
end
const _sim_exact_seeds = let rng = MersenneTwister(20250202)
    [rand(rng, UInt32) for s in 1:N_SIM, t in 2:inst.T]
end

function simulate_policy(m, is_ddu::Bool, dist::Symbol; n_paths::Int = N_SIM)
    @assert dist in (:exact, :saa) "dist must be :exact or :saa"
    profits = zeros(Float64, n_paths)
    x_hist  = zeros(Float64, n_paths, inst.T, nI)

    for s in 1:n_paths
        x_prev = copy(x0)
        total  = 0.0

        for t in 1:inst.T
            x_int  = [v > 0.5 ? 1 : 0 for v in x_prev]
            d_true = identify_region(inst, x_int)

            ξ = if t == 1
                nothing
            elseif dist == :saa
                Float64.(saa[d_true][_sim_saa_indices[s, t-1], :])
            else
                lrng = MersenneTwister(_sim_exact_seeds[s, t-1])
                za   = region_to_zone_active(inst, d_true)
                Float64[rand(lrng, _customer_dist(inst, j, za)) for j in 1:inst.num_customers]
            end

            if is_ddu
                model, _, θ_var, misc = get_or_build_ddu_model!(m, t, ξ, x_prev)
            else
                vf_next = t < inst.T ? m.V[t + 1] : ValueFn{Float64}()
                model, _, θ_var, misc = SDDPBAPE.get_or_build_model!(
                    m, t, vf_next, ξ, nothing, x_prev)
            end
            JuMP.optimize!(model)

            obj_val    = JuMP.objective_value(model)
            θ_val      = JuMP.value(θ_var)
            stage_prof = -(obj_val - θ_val)
            total     += stage_prof

            x_prev = Float64[JuMP.value(v) > 0.5 ? 1.0 : 0.0 for v in misc[:x_next]]
            x_hist[s, t, :] .= x_prev
        end

        profits[s] = total
    end

    return profits, x_hist
end

println("\nSimulating DDU-extra  — true distribution  ($N_SIM paths) ...")
profits_extra_true, x_hist_extra_true = simulate_policy(m_ddu_extra, true,  :exact)
println("Simulating DDU-extra  — SAA distribution   ($N_SIM paths) ...")
profits_extra_saa,  x_hist_extra_saa  = simulate_policy(m_ddu_extra, true,  :saa)
println("Simulating SDDiP      — true distribution  ($N_SIM paths) ...")
profits_std_true, x_hist_std_true     = simulate_policy(m_std, false, :exact)
println("Simulating SDDiP      — SAA distribution   ($N_SIM paths) ...")
profits_std_saa,  x_hist_std_saa      = simulate_policy(m_std, false, :saa)

# ── Part 4: Summary CSV ───────────────────────────────────────────────────────

out_dir = joinpath(@__DIR__, "..", "results", ITYPE)
mkpath(out_dir)
const FILE_TAG = "G2exp_$(ITYPE)_N$(N_SAA)"

summary_df = DataFrame(
    Instance_Type     = fill(ITYPE, 4),
    N_SAA             = fill(N_SAA, 4),
    Policy            = ["DDU-extra", "DDU-extra", "SDDiP",   "SDDiP"],
    Eval_Distribution = ["true",      "SAA",       "true",    "SAA"],
    Avg_Profit        = [mean(profits_extra_true), mean(profits_extra_saa),
                         mean(profits_std_true),  mean(profits_std_saa)],
    Std_Dev           = [std(profits_extra_true),  std(profits_extra_saa),
                         std(profits_std_true),   std(profits_std_saa)],
    First_Stage_Facs  = [string(opened_extra), string(opened_extra),
                         string(opened_std),   string(opened_std)],
    Active_Region     = [region1_extra, region1_extra, region1_std, region1_std],
    Final_LB          = [result_extra.lb,  result_extra.lb,  lb_std_final, lb_std_final],
    Iterations        = [result_extra.iters, result_extra.iters,
                         result_std.iters,   result_std.iters],
    Wall_Time_s       = [round(t_extra; digits=1), round(t_extra; digits=1),
                         round(t_std;   digits=1), round(t_std;   digits=1)],
    Explore_Regions   = [length(exploration_map), length(exploration_map), 0, 0],
)

CSV.write(joinpath(out_dir, "comparison_summary_$(FILE_TAG).csv"), summary_df)

println("\n" * "="^72)
println("SUMMARY")
println("="^72)
for row in eachrow(summary_df)
    @printf("%-12s | dist=%-5s | Avg profit: %9.2f | Std: %7.2f | Facs: %-12s | Region: %d\n",
        row.Policy, row.Eval_Distribution, row.Avg_Profit, row.Std_Dev,
        row.First_Stage_Facs, row.Active_Region)
end
println("="^72)

# ── Part 5: Convergence CSV ───────────────────────────────────────────────────

n_extra = length(extra_lb_hist)
n_std   = length(std_lb_hist)

conv_df = vcat(
    DataFrame(policy      = fill("DDU-extra", n_extra),
              iteration   = 1:n_extra,
              time_s      = extra_time_hist,
              lower_bound = extra_lb_hist),
    DataFrame(policy      = fill("SDDiP", n_std),
              iteration   = 1:n_std,
              time_s      = std_time_hist,
              lower_bound = std_lb_hist),
)
CSV.write(joinpath(out_dir, "convergence_data_$(FILE_TAG).csv"), conv_df)

# ── Part 6: Opening statistics CSVs ──────────────────────────────────────────

function compute_opening_freq(x_hist::Array{Float64,3},
                              policy_name::String, eval_dist::String)
    n_paths, T, nI_ = size(x_hist)
    rows = [(Policy            = policy_name,
             Eval_Distribution = eval_dist,
             Stage             = t,
             Facility          = i,
             Open_Frequency    = mean(x_hist[:, t, i]))
            for t in 1:T for i in 1:nI_]
    return DataFrame(rows)
end

function compute_opening_summary(x_hist::Array{Float64,3},
                                 policy_name::String, eval_dist::String)
    n_paths, T, nI_ = size(x_hist)
    rows = map(1:T) do t
        avg_n_open = mean(sum(x_hist[s, t, :]) for s in 1:n_paths)
        patterns   = [Tuple(Int.(x_hist[s, t, :])) for s in 1:n_paths]
        counts     = Dict{Tuple, Int}()
        for p in patterns
            counts[p] = get(counts, p, 0) + 1
        end
        modal_pat  = argmax(counts)
        modal_cnt  = counts[modal_pat]
        modal_facs = string(findall(==(1), collect(modal_pat)))
        modal_freq = modal_cnt / n_paths

        (Policy            = policy_name,
         Eval_Distribution = eval_dist,
         Stage             = t,
         Avg_N_Open        = round(avg_n_open; digits = 3),
         Modal_Facs        = modal_facs,
         Modal_Frequency   = round(modal_freq; digits = 3))
    end
    return DataFrame(rows)
end

open_freq_df = vcat(
    compute_opening_freq(x_hist_extra_true, "DDU-extra", "true"),
    compute_opening_freq(x_hist_extra_saa,  "DDU-extra", "SAA"),
    compute_opening_freq(x_hist_std_true,   "SDDiP",     "true"),
    compute_opening_freq(x_hist_std_saa,    "SDDiP",     "SAA"),
)
open_summ_df = vcat(
    compute_opening_summary(x_hist_extra_true, "DDU-extra", "true"),
    compute_opening_summary(x_hist_extra_saa,  "DDU-extra", "SAA"),
    compute_opening_summary(x_hist_std_true,   "SDDiP",     "true"),
    compute_opening_summary(x_hist_std_saa,    "SDDiP",     "SAA"),
)

CSV.write(joinpath(out_dir, "opening_stats_$(FILE_TAG).csv"),   open_freq_df)
CSV.write(joinpath(out_dir, "opening_summary_$(FILE_TAG).csv"), open_summ_df)

# ── Part 7: Convergence plots ─────────────────────────────────────────────────

gr()

p1 = plot(1:n_extra, extra_lb_hist;
    label     = "DDU-extra",
    xlabel    = "Iteration",
    ylabel    = "Lower Bound",
    title     = "Iteration vs Lower Bound",
    linewidth = 2,
    color     = :blue,
    legend    = :bottomright,
)
plot!(p1, 1:n_std, std_lb_hist;
    label     = "SDDiP",
    linewidth = 2,
    color     = :red,
    linestyle = :dash,
)

p2 = plot(extra_time_hist, extra_lb_hist;
    label     = "DDU-extra",
    xlabel    = "Wall time (s)",
    ylabel    = "Lower Bound",
    title     = "Time vs Lower Bound",
    linewidth = 2,
    color     = :blue,
    legend    = :bottomright,
)
plot!(p2, std_time_hist, std_lb_hist;
    label     = "SDDiP",
    linewidth = 2,
    color     = :red,
    linestyle = :dash,
)

fig_conv = plot(p1, p2; layout = (1, 2), size = (1200, 450), margin = 5Plots.mm)
savefig(fig_conv, joinpath(out_dir, "convergence_plot_$(FILE_TAG).png"))

# ── Part 8: Opening-frequency heatmaps ───────────────────────────────────────

function opening_heatmap(x_hist, title_str)
    n, T, nI_ = size(x_hist)
    freq_mat = [mean(x_hist[:, t, i]) for i in 1:nI_, t in 1:T]
    heatmap(1:T, 1:nI_, freq_mat;
        xlabel = "Stage",
        ylabel = "Facility",
        title  = title_str,
        clims  = (0.0, 1.0),
        color  = :Blues,
        yticks = 1:nI_,
        xticks = 1:T,
    )
end

p_extra_true = opening_heatmap(x_hist_extra_true, "DDU-extra / true dist")
p_extra_saa  = opening_heatmap(x_hist_extra_saa,  "DDU-extra / SAA dist")
p_std_true   = opening_heatmap(x_hist_std_true,   "SDDiP / true dist")
p_std_saa    = opening_heatmap(x_hist_std_saa,    "SDDiP / SAA dist")

fig_open = plot(p_extra_true, p_extra_saa, p_std_true, p_std_saa;
    layout = (2, 2), size = (1200, 800), margin = 5Plots.mm)
savefig(fig_open, joinpath(out_dir, "opening_freq_plot_$(FILE_TAG).png"))

println("\nFiles written to: $out_dir")
println("  comparison_summary_$(FILE_TAG).csv")
println("  convergence_data_$(FILE_TAG).csv")
println("  opening_stats_$(FILE_TAG).csv")
println("  opening_summary_$(FILE_TAG).csv")
println("  convergence_plot_$(FILE_TAG).png")
println("  opening_freq_plot_$(FILE_TAG).png")
