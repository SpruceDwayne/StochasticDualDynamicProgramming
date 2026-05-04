"""
run_complexity.jl
=================
Single-configuration DDU-SDDiP runner for the computational complexity study.
Handles any instance size and any number of stages via dynamic Stage construction.

Usage
-----
    julia run_complexity.jl <inst_json> <N_SAA> <max_iter> <time_budget_s> <seed> <out_dir> <tag>

Arguments
---------
  inst_json      : path to the instance JSON file
  N_SAA          : scenarios per DDU region (SAA discretization)
  max_iter       : hard iteration cap
  time_budget_s  : wall-time budget in seconds (soft stop inside logfn)
  seed           : integer seed for SAA sampling and forward-pass RNG
  out_dir        : directory for output files (created if absent)
  tag            : string identifier embedded in all output filenames

Outputs (written to out_dir/)
------------------------------
  convergence_<tag>.csv  -- one row per iteration:
      iteration, time_s, iter_time_s, lb, delta_lb, cuts_total, cuts_new
  summary_<tag>.json     -- scalar summary of the run
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "SDDPBAPE"))
for pkg in ["CSV", "DataFrames", "JSON"]
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end

using SDDPBAPE
using JuMP, Gurobi
using Printf, Statistics, Random
using CSV, DataFrames, JSON

# Gurobi environment: suppress solver output
const GRB_ENV = Gurobi.Env(output_flag = 0)

# Include shared problem helpers (distributions, data loader, identify_region)
include(joinpath(@__DIR__, "..", "problem_src", "distributions.jl"))

# ── Exception used for time-budget early stopping ─────────────────────────────
struct TimeBudgetExceeded <: Exception end

# ── Stage builder (complexity-study version) ───────────────────────────────────
# Identical in structure to make_stage_builder in problem_src/subproblem_G.jl
# but with theta_lb and M_BIG derived from instance parameters so they remain
# valid across all instance sizes tested in the scaling experiments.
function make_stage_builder_cx(
    inst::InstanceData, t::Int, grb_env::Gurobi.Env, max_stage_profit::Float64
)
    nI_  = inst.num_facilities
    nJ_  = inst.num_customers
    nZ_  = inst.num_zones
    nD_  = length(inst.activation_regions)
    P_   = inst.profit_matrix
    O_   = Float64(inst.O)
    C_   = Float64(inst.C)
    k_   = inst.k
    is_root_     = (t == 1)
    is_terminal_ = (t == inst.T)
    # Lower bound on θ: worst-case future profit summed over remaining stages.
    theta_lb_ = is_terminal_ ? 0.0 : -Float64(inst.T - t) * max_stage_profit

    function builder(_t, _vf, ω; fix_state)
        model = Model(() -> Gurobi.Optimizer(grb_env))
        set_silent(model)

        # ── Facility decisions ────────────────────────────────────────────────
        @variable(model, x[1:nI_], Bin)

        # ── Entering-state copy variables (non-root stages) ───────────────────
        z_refs = JuMP.VariableRef[]
        if !is_root_
            @variable(model, z[1:nI_], lower_bound = 0.0, upper_bound = 1.0)
            for i in 1:nI_
                @constraint(model, z[i] == fix_state[i])
            end
            append!(z_refs, z)

            # Monotonicity: can only open, not close
            for i in 1:nI_
                @constraint(model, x[i] >= z_refs[i])
            end
            # Per-stage opening budget
            @constraint(model, sum(x[i] - z_refs[i] for i in 1:nI_) <= k_)

            # ── Demand allocation ─────────────────────────────────────────────
            demand = Float64[ω[j] for j in 1:nJ_]
            @variable(model, w[1:nI_, 1:nJ_] >= 0)
            for j in 1:nJ_
                @constraint(model, sum(w[i, j] for i in 1:nI_) <= demand[j])
            end
            for i in 1:nI_
                @constraint(model, sum(w[i, j] for j in 1:nJ_) <= C_ * z_refs[i])
            end
        else
            # Stage 1: pure investment, no demand revenue
            @constraint(model, sum(x) <= k_)
        end

        # ── Zone activation + region indicator variables (non-terminal) ────────
        region_ind = Dict{Int, JuMP.VariableRef}()
        if !is_terminal_
            @variable(model, a[1:nZ_], Bin)
            @variable(model, δ[1:nD_], Bin)

            for zz in 1:nZ_
                fz = inst.zone_facilities[zz]
                @constraint(model, a[zz] <= sum(x[i] for i in fz))
                for i in fz
                    @constraint(model, a[zz] >= x[i])
                end
            end

            @constraint(model, sum(δ) == 1)
            for zz in 1:nZ_
                @constraint(model,
                    a[zz] == sum(δ[d] for d in 1:nD_ if zz in inst.activation_regions[d]))
            end

            for d in 1:nD_
                region_ind[d] = δ[d]
            end
        end

        # ── Continuation variable ─────────────────────────────────────────────
        if is_terminal_
            @variable(model, θ == 0.0)
        else
            @variable(model, θ >= theta_lb_)
        end

        # ── Objective ─────────────────────────────────────────────────────────
        if is_root_
            @objective(model, Min, O_ * sum(x) + θ)
        else
            opening = O_ * sum(x[i] - z_refs[i] for i in 1:nI_)
            @objective(model, Min,
                -sum(P_[i, j] * w[i, j] for i in 1:nI_, j in 1:nJ_) + opening + θ)
        end

        misc = Dict{Symbol, Any}(
            :x_next            => JuMP.VariableRef[x[i] for i in 1:nI_],
            :region_indicators => region_ind,
        )
        if !is_root_
            misc[:z_vars] = z_refs
        end
        return model, z_refs, θ, misc
    end

    return builder
end

# ── Main run function ──────────────────────────────────────────────────────────
function main()
    length(ARGS) >= 7 || error(
        "Usage: julia run_complexity.jl " *
        "<inst_json> <N_SAA> <max_iter> <time_budget_s> <seed> <out_dir> <tag>"
    )

    inst_path   = ARGS[1]
    N_SAA       = parse(Int,     ARGS[2])
    max_iter    = parse(Int,     ARGS[3])
    time_budget = parse(Float64, ARGS[4])
    saa_seed    = parse(Int,     ARGS[5])
    out_dir     = ARGS[6]
    run_tag     = ARGS[7]

    mkpath(out_dir)

    # ── Load instance ──────────────────────────────────────────────────────────
    inst      = load_instance(inst_path)
    nI        = inst.num_facilities
    nJ        = inst.num_customers
    nZ        = inst.num_zones
    n_regions = length(inst.activation_regions)
    x0        = zeros(Float64, nI)
    ζ_init    = 1

    # Scaling: upper bound on per-stage profit used for theta_lb and M_BIG.
    # max_stage_profit = full demand × revenue per unit, for all customers.
    max_stage_profit = Float64(nJ) * Float64(inst.max_demand) * Float64(inst.R)
    M_BIG_val        = Float64(inst.T) * max_stage_profit * 2.0

    @printf("Instance : %s\n", basename(inst_path))
    @printf("  I=%-3d J=%-3d Z=%-2d T=%-2d regions=%-4d k=%d\n",
            nI, nJ, nZ, inst.T, n_regions, inst.k)
    @printf("  N_SAA=%-4d max_iter=%-4d budget=%.0fs seed=%d\n",
            N_SAA, max_iter, time_budget, saa_seed)
    @printf("  M_BIG=%.2e  max_stage_profit=%.0f\n", M_BIG_val, max_stage_profit)

    # ── Setup: SAA + regions ───────────────────────────────────────────────────
    t_setup_start = time()

    saa       = generate_saa_scenarios(inst; N = N_SAA, seed = saa_seed)
    uniform_p = fill(1.0 / N_SAA, N_SAA)
    ddu_regs  = DDURegion[
        DDURegion(d, Any[saa[d][n, :] for n in 1:N_SAA], uniform_p)
        for d in 1:n_regions
    ]

    unit_weight = _ -> 1.0
    noop_ctx    = (t, ζ, ω) -> ζ
    noop_key    = x -> x

    # ── Dynamic Stage[] construction ───────────────────────────────────────────
    # Stage 1: sampler returns nothing (pure investment, no demand).
    # Stages 2:T: sampler draws uniformly from the N_SAA scenarios of the
    #             incoming region ζ. The rng is advanced in call order.
    rng_fw = MersenneTwister(saa_seed + 1)

    stages_ddu = Stage[]
    for t in 1:inst.T
        builder = make_stage_builder_cx(inst, t, GRB_ENV, max_stage_profit)
        stg = if t == 1
            Stage(1, nI, builder,
                  (ζ) -> nothing, unit_weight,
                  (tt, ζ) -> (ddu_regs[ζ].Xi, ddu_regs[ζ].pXi), noop_ctx, noop_key)
        else
            Stage(t, nI, builder,
                  (ζ) -> ddu_regs[ζ].Xi[rand(rng_fw, 1:N_SAA)], unit_weight,
                  (tt, ζ) -> (ddu_regs[ζ].Xi, ddu_regs[ζ].pXi), noop_ctx, noop_key)
        end
        push!(stages_ddu, stg)
    end

    regions_per_stage = Vector{Vector{DDURegion}}(undef, inst.T)
    for t in 1:(inst.T - 1)
        regions_per_stage[t] = ddu_regs
    end
    regions_per_stage[inst.T] = DDURegion[]

    #m_ddu = DDUSDDP(stages_ddu, regions_per_stage; M_big = M_BIG_val, discount = 1.0)
    # big-M (default, works with any solver)
    m_ddu = DDUSDDP(stages_ddu, regions_per_stage; M_big = M_BIG_val, discount = 1.0)

# indicator constraints (Gurobi/CPLEX only — M_big ignored)
m_ddu = DDUSDDP(stages_ddu, regions_per_stage; use_indicators = true, discount = 1.0)

    config = SDDiPConfig(
        cut_type        = :lagrangian,
        burnin_iters    = 0,
        burnin_cut_type = :IO,
        level_cfg       = LevelMethodConfig(optimizer = () -> Gurobi.Optimizer(GRB_ENV)),
    )

    t_setup = time() - t_setup_start
    @printf("Setup complete in %.1f s\n\n", t_setup)

    # ── Training loop with per-iteration timing ────────────────────────────────
    lb_hist         = Float64[]
    time_hist       = Float64[]
    iter_time_hist  = Float64[]
    cuts_total_hist = Int[]
    cuts_new_hist   = Int[]
    delta_lb_hist   = Float64[]

    t_run_start = time()
    prev_time   = Ref(t_run_start)
    stop_reason = Ref("max_iter")
    converged   = Ref(false)

    logfn = function (it, stats)
        now_t     = time()
        elapsed   = now_t - t_run_start
        iter_time = now_t - prev_time[]
        prev_time[] = now_t

        push!(lb_hist,         stats.lb)
        push!(time_hist,       elapsed)
        push!(iter_time_hist,  iter_time)
        push!(cuts_total_hist, stats.total_cuts)
        push!(cuts_new_hist,   stats.new_cuts)
        push!(delta_lb_hist,   stats.Δlb)

        @printf("iter %4d | LB=%12.4f | ΔLB=%9.3e | cuts %3d/%-4d | %7.1fs | iter %.2fs\n",
                it, stats.lb, stats.Δlb, stats.new_cuts, stats.total_cuts,
                elapsed, iter_time)

        if elapsed >= time_budget
            stop_reason[] = "time_budget"
            throw(TimeBudgetExceeded())
        end
    end

    println("="^76)
    @printf("DDU-SDDiP  |  %s\n", run_tag)
    println("="^76)

    try
        r = run_ddu_sddip!(m_ddu;
            x0          = x0,
            ζ_init      = ζ_init,
            config      = config,
            S           = 1,
            max_iter    = max_iter,
            patience    = 10,
            force_every = 2,
            cut_atol    = 1e-6,
            logfn       = logfn,
        )
        # run_ddu_sddip! returns early on patience → iters < max_iter
        if r.iters < max_iter
            stop_reason[] = "patience"
            converged[]   = true
        end
    catch e
        e isa TimeBudgetExceeded || rethrow(e)
    end

    t_total = time() - t_run_start
    n_iters = length(lb_hist)
    final_lb = isempty(lb_hist) ? NaN : last(lb_hist)

    println("="^76)
    @printf("Done: %d iters | stop=%s | LB=%.4f | wall=%.1fs | setup=%.1fs\n",
            n_iters, stop_reason[], final_lb, t_total, t_setup)
    println("="^76)

    # ── Write convergence CSV ──────────────────────────────────────────────────
    conv_df = DataFrame(
        iteration   = 1:n_iters,
        time_s      = time_hist,
        iter_time_s = iter_time_hist,
        lb          = lb_hist,
        delta_lb    = delta_lb_hist,
        cuts_total  = cuts_total_hist,
        cuts_new    = cuts_new_hist,
    )
    conv_path = joinpath(out_dir, "convergence_$(run_tag).csv")
    CSV.write(conv_path, conv_df)

    # ── Write summary JSON ─────────────────────────────────────────────────────
    summary = Dict{String, Any}(
        "tag"              => run_tag,
        "inst_path"        => inst_path,
        "N_SAA"            => N_SAA,
        "nI"               => nI,
        "nJ"               => nJ,
        "nZ"               => nZ,
        "n_regions"        => n_regions,
        "T"                => inst.T,
        "k"                => inst.k,
        "seed"             => saa_seed,
        "converged"        => converged[],
        "stop_reason"      => stop_reason[],
        "final_lb"         => isnan(final_lb) ? nothing : final_lb,
        "total_time_s"     => t_total,
        "total_iters"      => n_iters,
        "mean_iter_time_s" => n_iters > 0  ? mean(iter_time_hist) : nothing,
        "std_iter_time_s"  => n_iters > 1  ? std(iter_time_hist)  : nothing,
        "setup_time_s"     => t_setup,
        "M_BIG"            => M_BIG_val,
    )
    summary_path = joinpath(out_dir, "summary_$(run_tag).json")
    open(summary_path, "w") do io
        JSON.print(io, summary, 2)
    end

    println("\nOutput written to: $out_dir")
    @printf("  convergence_%s.csv   (%d rows)\n", run_tag, n_iters)
    @printf("  summary_%s.json\n", run_tag)
end

main()
