import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "SDDPBAPE"))
for pkg in ["Statistics"]
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end

using SDDPBAPE
using JuMP, Gurobi
const GRB_ENV = Gurobi.Env(output_flag = 0)
using Printf, Statistics, Random

include(joinpath(@__DIR__, "distributions.jl"))
include(joinpath(@__DIR__, "subproblem_G.jl"))

# ── CLI args (same as run_comparison_G2.jl) ───────────────────────────────────
const ITYPE = length(ARGS) >= 1 ? uppercase(strip(ARGS[1])) : "D"
@assert ITYPE in ("A","B","C","D") "Unknown type '$ITYPE'"
const N_SAA = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 25

const INST_PATH = joinpath(@__DIR__, "..", "..", "instance_data_$(ITYPE).json")
const inst      = load_instance(INST_PATH)
@printf("Smoketest | type %s | %d facs | %d customers | %d zones | T=%d\n",
    inst.interaction_type, inst.num_facilities, inst.num_customers, inst.num_zones, inst.T)

const nI        = inst.num_facilities
const n_regions = length(inst.activation_regions)
const saa       = generate_saa_scenarios(inst; N=N_SAA, seed=123)
const uniform_p = fill(1.0 / N_SAA, N_SAA)
const x0        = zeros(Float64, nI)
const ζ_init    = 1
const M_BIG     = 1e5

const N_SMOKE   = 200   # small enough to be fast, large enough for diagnostics

const ddu_regions = DDURegion[
    DDURegion(d, Any[saa[d][n, :] for n in 1:N_SAA], uniform_p)
    for d in 1:n_regions
]

const unit_weight = _ -> 1.0
const noop_ctx    = (t, ζ, ω) -> ζ
const noop_key    = x -> x

# ── Stage builders (identical to run_comparison_G2.jl) ───────────────────────

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

# ── Models (same parameters as run_comparison_G2.jl) ─────────────────────────

regions_per_stage = Vector{Vector{DDURegion}}(undef, inst.T)
for t in 1:inst.T - 1
    regions_per_stage[t] = ddu_regions
end
regions_per_stage[inst.T] = DDURegion[]

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

m_ddu = DDUSDDP(stages_ddu, regions_per_stage; M_big = M_BIG, discount = 1.0)

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

# ── Reduced training ──────────────────────────────────────────────────────────
config = SDDiPConfig(
    cut_type        = :lagrangian,
    burnin_iters    = 0,
    burnin_cut_type = :IO,
    level_cfg       = LevelMethodConfig(optimizer = () -> Gurobi.Optimizer(GRB_ENV)),
)

println("\n" * "="^60)
println("Training DDU-SDDiP (reduced: max_iter=50, patience=10) ...")
println("="^60)
result_ddu = run_ddu_sddip!(m_ddu;
    x0 = x0, ζ_init = ζ_init, config = config,
    S = 1, max_iter = 50, patience = 10, force_every = 2, cut_atol = 1e-6)
@printf("DDU done: %d iters | LB=%.4f\n", result_ddu.iters, result_ddu.lb)

ω1_ddu = m_ddu.stages[1].sampler(ζ_init)
model1_ddu, _, _, misc1_ddu = get_or_build_ddu_model!(m_ddu, 1, ω1_ddu, x0)
JuMP.optimize!(model1_ddu)
x1_ddu_int  = [JuMP.value(v) > 0.5 ? 1 : 0 for v in misc1_ddu[:x_next]]
opened_ddu  = findall(==(1), x1_ddu_int)
region1_ddu = identify_region(inst, x1_ddu_int)
@printf("  DDU stage-1: facs=%s  region=%d\n", string(opened_ddu), region1_ddu)

println("\n" * "="^60)
println("Training SDDiP (reduced: max_iter=50, patience=10) ...")
println("="^60)
result_std = run_sddip!(m_std;
    x0 = x0, ctx0 = nothing, config = config,
    S = 1, max_iter = 50, patience = 10, force_every = 2, cut_atol = 1e-6)

vf2_std    = m_std.V[2]
model1_std, _, _, misc1_std = SDDPBAPE.get_or_build_model!(
    m_std, 1, vf2_std, nothing, nothing, x0)
JuMP.optimize!(model1_std)
x1_std_int  = [JuMP.value(v) > 0.5 ? 1 : 0 for v in misc1_std[:x_next]]
opened_std  = findall(==(1), x1_std_int)
region1_std = identify_region(inst, x1_std_int)
lb_std      = JuMP.objective_value(model1_std)
@printf("STD done: %d iters | LB=%.4f\n", result_std.iters, lb_std)
@printf("  STD stage-1: facs=%s  region=%d\n", string(opened_std), region1_std)

# ── Pre-generated scenario draws (same for both models per dist) ──────────────
const smoke_saa_idx   = let rng = MersenneTwister(20250101)
    [rand(rng, 1:N_SAA) for s in 1:N_SMOKE, t in 2:inst.T]
end
const smoke_ex_seeds  = let rng = MersenneTwister(20250202)
    [rand(rng, UInt32) for s in 1:N_SMOKE, t in 2:inst.T]
end

# ── Verbose simulation: records per-(path,stage) details ─────────────────────
struct StageRecord
    t::Int
    d_true::Int
    demand_sum::Float64
    capacity_avail::Float64   # C × #open facilities entering this stage (0 at t=1)
    x_prev::Vector{Int}
    x_next::Vector{Int}
    obj_val::Float64
    θ_val::Float64
    stage_prof::Float64
end

function simulate_policy_verbose(m, is_ddu::Bool, dist::Symbol;
                                  n_paths::Int = N_SMOKE)
    @assert dist in (:exact, :saa)
    profits = zeros(Float64, n_paths)
    x_hist  = zeros(Float64, n_paths, inst.T, nI)
    records = [StageRecord[] for _ in 1:n_paths]

    for s in 1:n_paths
        x_prev = copy(x0)
        total  = 0.0

        for t in 1:inst.T
            x_int  = [v > 0.5 ? 1 : 0 for v in x_prev]
            d_true = identify_region(inst, x_int)

            ξ = if t == 1
                nothing
            elseif dist == :saa
                Float64.(saa[d_true][smoke_saa_idx[s, t-1], :])
            else
                lrng = MersenneTwister(smoke_ex_seeds[s, t-1])
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

            x_next = [JuMP.value(v) > 0.5 ? 1 : 0 for v in misc[:x_next]]
            # Capacity available at this stage = C × open facilities entering the stage.
            # At t=1 no demand is served (investment only) so capacity is 0 by convention.
            cap = t == 1 ? 0.0 : inst.C * Float64(sum(x_int))
            push!(records[s], StageRecord(
                t, d_true,
                ξ === nothing ? 0.0 : sum(ξ),
                cap,
                Int.(round.(x_prev)),
                x_next,
                obj_val, θ_val, stage_prof,
            ))

            x_prev = Float64.(x_next)
            x_hist[s, t, :] .= x_next
        end

        profits[s] = total
    end

    return profits, x_hist, records
end

# ── Diagnostics ───────────────────────────────────────────────────────────────
function print_analysis(profits, records, label)
    n = length(profits)
    println("\n" * "="^60)
    println("ANALYSIS: $label  (n=$n paths)")
    println("="^60)

    μ, σ = mean(profits), std(profits)
    @printf("  Profit:  mean=%10.2f  std=%10.4f  min=%10.2f  max=%10.2f\n",
        μ, σ, minimum(profits), maximum(profits))

    if σ < 1e-4 * (abs(μ) + 1.0)
        println("  *** WARNING: std_dev is essentially zero relative to mean ***")
    end

    qs = quantile(profits, [0.05, 0.25, 0.50, 0.75, 0.95])
    @printf("  Quantiles [5,25,50,75,95]%%: %s\n",
        join([@sprintf("%.1f", q) for q in qs], "  "))

    # Per-stage profit breakdown + capacity-binding check
    println("\n  Per-stage profit:")
    for t in 1:inst.T
        sp  = [records[s][t].stage_prof    for s in 1:n]
        cap = [records[s][t].capacity_avail for s in 1:n]
        dem = [records[s][t].demand_sum     for s in 1:n]
        @printf("    t=%d  mean=%9.2f  std=%9.4f  range=[%9.2f, %9.2f]\n",
            t, mean(sp), std(sp), minimum(sp), maximum(sp))
        if std(sp) < 1e-4 * (abs(mean(sp)) + 1.0)
            println("         ^^^ stage $t has near-zero profit variance")
            if t > 1
                # Check whether demand >= capacity (capacity constraints always binding).
                # When true, revenue = C × #open × avg_profit_rate, independent of demand,
                # so profit has zero variance even though demand varies.
                n_binding     = sum(dem[s] >= cap[s] for s in 1:n)
                cap_val       = cap[1]  # same for all paths if x pattern is fixed
                @printf("         Capacity check (demand ≥ C×#open_facs):  %d/%d paths (%.0f%%)\n",
                    n_binding, n, 100.0 * n_binding / n)
                @printf("         C×#open = %.1f  |  demand_sum: mean=%.2f  min=%.2f  max=%.2f\n",
                    cap_val, mean(dem), minimum(dem), maximum(dem))
                if n_binding == n
                    println("         => Capacity ALWAYS binding: revenue is constant regardless of demand.")
                    println("            This explains std≈0. Consider increasing C or reducing demand scale.")
                elseif n_binding > n ÷ 2
                    println("         => Capacity binding in majority of paths — partial explanation for low std.")
                end
            end
        end
    end

    # Per-stage (obj - θ): should equal -(stage_prof), i.e., the current-stage cost
    println("\n  Per-stage (obj_val − θ_val)  [= negative current-stage profit]:")
    for t in 1:inst.T
        diffs = [records[s][t].obj_val - records[s][t].θ_val for s in 1:n]
        @printf("    t=%d  mean=%9.2f  std=%9.4f\n", t, mean(diffs), std(diffs))
    end

    # Per-stage x_next decision patterns
    println("\n  Per-stage x_next facility patterns:")
    for t in 1:inst.T
        pats = Dict{Vector{Int}, Int}()
        for s in 1:n
            p = records[s][t].x_next
            pats[p] = get(pats, p, 0) + 1
        end
        @printf("    t=%d  %d distinct x_next patterns\n", t, length(pats))
        for (pat, cnt) in sort(collect(pats), by = kv -> -kv[2])
            facs = findall(==(1), pat)
            @printf("      facs=%-20s  %4d/%d paths\n", string(facs), cnt, n)
        end
    end

    # Per-stage demand and region
    println("\n  Per-stage demand totals and regions:")
    for t in 2:inst.T
        ds  = [records[s][t].demand_sum for s in 1:n]
        rgs = [records[s][t].d_true     for s in 1:n]
        unique_rgs = sort(collect(Set(rgs)))
        @printf("    t=%d  demand_sum: mean=%.2f  std=%.4f  regions=%s\n",
            t, mean(ds), std(ds), string(unique_rgs))
        if length(unique_rgs) == 1
            println("         ^^^ single region throughout — demand distribution fixed")
        end
        if std(ds) < 1e-4
            println("         ^^^ demand_sum is constant — all paths see same scenario")
        end
    end

    # Flag if decisions never change (policy locks in after stage 1)
    all_same_x = true
    for t in 2:inst.T
        x_ref = records[1][t].x_next
        for s in 2:n
            if records[s][t].x_next != x_ref
                all_same_x = false
                break
            end
        end
        all_same_x || break
    end
    if all_same_x
        println("\n  *** All paths share the same x_next at every stage — policy is deterministic ***")
        println("      Profit variance comes solely from demand (revenue) variation.")
        println("      If std≈0, check: is capacity always fully utilized regardless of demand?")
    end
end

# ── Run simulations and print diagnostics ─────────────────────────────────────
for (dist, label_prefix) in [(:exact, "true dist"), (:saa, "SAA dist")]
    println("\n" * "="^72)
    @printf("Simulating DDU-SDDiP — %s (%d paths) ...\n", label_prefix, N_SMOKE)
    p_ddu, xh_ddu, rec_ddu = simulate_policy_verbose(m_ddu, true, dist)
    print_analysis(p_ddu, rec_ddu, "DDU-SDDiP / $label_prefix")

    @printf("\nSimulating SDDiP     — %s (%d paths) ...\n", label_prefix, N_SMOKE)
    p_std, xh_std, rec_std = simulate_policy_verbose(m_std, false, dist)
    print_analysis(p_std, rec_std, "SDDiP / $label_prefix")

    # Cross-model comparison for same dist (same scenario draws)
    println("\n  --- Cross-model comparison (same scenario paths, $label_prefix) ---")
    diffs = p_ddu .- p_std
    @printf("  DDU − STD profit:  mean=%9.2f  std=%9.4f  min=%9.2f  max=%9.2f\n",
        mean(diffs), std(diffs), minimum(diffs), maximum(diffs))
    pct_ddu_wins = 100 * mean(p_ddu .> p_std)
    @printf("  DDU better in %.1f%% of paths\n", pct_ddu_wins)
end

println("\n" * "="^72)
println("Smoketest complete.")
