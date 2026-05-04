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
# Usage: julia run_comparison_G2.jl [TYPE] [N_SAA]
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
const M_BIG     = 1e5

const ddu_regions = DDURegion[
    DDURegion(d, Any[saa[d][n, :] for n in 1:N_SAA], uniform_p)
    for d in 1:n_regions
]

# Stage-1 deterministic demand (expected under region 1)
const zone_active_0 = region_to_zone_active(inst, 1)
const pmfs_region1  = compute_demand_distributions(inst, zone_active_0)
const expected_d1   = Float64[
    sum(Float64(k) * pmfs_region1[j][k+1] for k in 0:inst.max_demand)
    for j in 1:inst.num_customers
]

const unit_weight = _ -> 1.0
const noop_ctx    = (t, ζ, ω) -> ζ
const noop_key    = x -> x

# ── Exogenous stage builder (no DDU variables) ────────────────────────────────
# Used for the SDDiP baseline, which ignores decision-dependent uncertainty.
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

# ── Part 1: DDU-SDDiP ─────────────────────────────────────────────────────────

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

m_ddu = DDUSDDP(stages_ddu, regions_per_stage; M_big = M_BIG, discount = 1.0)

config = SDDiPConfig(
    cut_type        = :lagrangian,
    burnin_iters    = 0,
    burnin_cut_type = :IO,
    level_cfg       = LevelMethodConfig(optimizer = () -> Gurobi.Optimizer(GRB_ENV)),
)

ddu_lb_hist   = Float64[]
ddu_time_hist = Float64[]
t_ddu_start   = time()

function ddu_logfn(it, stats)
    push!(ddu_lb_hist,   stats.lb)
    push!(ddu_time_hist, time() - t_ddu_start)
    @printf("DDU-SDDiP iter %4d | new cuts: %2d | total: %3d | LB=%.4f | ΔLB=%.3e\n",
        it, stats.new_cuts, stats.total_cuts, stats.lb, stats.Δlb)
end

println("\n" * "="^72)
println("DDU-SDDiP")
println("="^72)

result_ddu = run_ddu_sddip!(m_ddu;
    x0          = x0,
    ζ_init      = ζ_init,
    config      = config,
    S           = 1,
    max_iter    = 150,
    patience    = 10,
    force_every = 2,
    cut_atol    = 1e-6,
    logfn       = ddu_logfn,
)
t_ddu = time() - t_ddu_start

# Extract DDU-SDDiP first-stage decision
ω1_ddu = m_ddu.stages[1].sampler(ζ_init)
model1_ddu, _, _, misc1_ddu = get_or_build_ddu_model!(m_ddu, 1, ω1_ddu, x0)
JuMP.optimize!(model1_ddu)
x1_ddu_vals = JuMP.value.(misc1_ddu[:x_next])
x1_ddu_int  = [v > 0.5 ? 1 : 0 for v in x1_ddu_vals]
opened_ddu  = findall(v -> v > 0.5, x1_ddu_vals)
region1_ddu = identify_region(inst, x1_ddu_int)

@printf("\nDDU-SDDiP done: %d iters | LB=%.4f | %.1f s\n",
    result_ddu.iters, result_ddu.lb, t_ddu)
@printf("  Facilities opened: %s | region d=%d\n", string(opened_ddu), region1_ddu)

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
ω1_std        = nothing   # stage 1 is investment-only; no demand scenario needed

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
    config      = config,
    S           = 1,
    max_iter    = 150,
    patience    = 10,
    force_every = 2,
    cut_atol    = 1e-6,
    logfn       = std_logfn,
)
t_std = time() - t_std_start

# Extract SDDiP first-stage decision
vf2_std     = m_std.V[2]
model1_std, _, _, misc1_std = SDDPBAPE.get_or_build_model!(m_std, 1, vf2_std, ω1_std, nothing, x0)
JuMP.optimize!(model1_std)
x1_std_vals = JuMP.value.(misc1_std[:x_next])
x1_std_int  = [v > 0.5 ? 1 : 0 for v in x1_std_vals]
opened_std  = findall(v -> v > 0.5, x1_std_vals)
region1_std = identify_region(inst, x1_std_int)
lb_std_final = JuMP.objective_value(model1_std)

@printf("\nSDDiP done: %d iters | LB=%.4f | %.1f s\n",
    result_std.iters, lb_std_final, t_std)
@printf("  Facilities opened: %s | region d=%d\n", string(opened_std), region1_std)

# ── Part 3: Out-of-sample simulation ─────────────────────────────────────────
# Each policy is evaluated under two distributions:
#   :true — demand for the realised region d sampled from the true BetaBinomial PMF
#   :saa  — demand for the realised region d sampled uniformly from the SAA scenarios
#
# In both cases the region d is determined by the policy's actual post-decision
# facility vector x_{t-1}, so the evaluation remains faithful to the DDU structure.
# Per-path decisions (x_hist[s, t, i]) are recorded for all stages.

const N_SIM = 1000

function simulate_policy(m, is_ddu::Bool, dist::Symbol;
                         n_paths::Int = N_SIM, seed::Int = 1111)
    @assert dist in (:exact, :saa) "dist must be :exact or :saa"
    rng     = MersenneTwister(seed)
    profits = zeros(Float64, n_paths)
    x_hist  = zeros(Float64, n_paths, inst.T, nI)

    for s in 1:n_paths
        x_prev = copy(x0)
        total  = 0.0

        for t in 1:inst.T
            # Region is always determined by the actual facility decisions
            x_int  = [v > 0.5 ? 1 : 0 for v in x_prev]
            d_true = identify_region(inst, x_int)

            ξ = if t == 1
                nothing  # stage 1 is a pure investment stage
            elseif dist == :exact
                za = region_to_zone_active(inst, d_true)
                Float64[rand(rng, _customer_dist(inst, j, za)) for j in 1:inst.num_customers]
            else  # :saa
                Float64.(saa[d_true][rand(rng, 1:N_SAA), :])
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

println("\nSimulating DDU-SDDiP — true distribution  ($N_SIM paths) ...")
profits_ddu_true, x_hist_ddu_true = simulate_policy(m_ddu, true,  :exact; seed = 1111)
println("Simulating DDU-SDDiP — SAA distribution   ($N_SIM paths) ...")
profits_ddu_saa,  x_hist_ddu_saa  = simulate_policy(m_ddu, true,  :saa;  seed = 2222)
println("Simulating SDDiP     — true distribution  ($N_SIM paths) ...")
profits_std_true, x_hist_std_true = simulate_policy(m_std, false, :exact; seed = 3333)
println("Simulating SDDiP     — SAA distribution   ($N_SIM paths) ...")
profits_std_saa,  x_hist_std_saa  = simulate_policy(m_std, false, :saa;  seed = 4444)

# ── Part 4: Summary CSV ───────────────────────────────────────────────────────
# Four rows: one per (policy, evaluation distribution) combination.
# Training metadata (LB, iterations, wall time) is identical across the two
# evaluation distributions for the same policy.

out_dir = joinpath(@__DIR__, "..", "results", ITYPE)
mkpath(out_dir)
const FILE_TAG = "G2_$(ITYPE)_N$(N_SAA)"

summary_df = DataFrame(
    Instance_Type     = fill(ITYPE, 4),
    N_SAA             = fill(N_SAA, 4),
    Policy            = ["DDU-SDDiP", "DDU-SDDiP", "SDDiP",    "SDDiP"],
    Eval_Distribution = ["true",      "SAA",       "true",     "SAA"],
    Avg_Profit        = [mean(profits_ddu_true), mean(profits_ddu_saa),
                         mean(profits_std_true), mean(profits_std_saa)],
    Std_Dev           = [std(profits_ddu_true),  std(profits_ddu_saa),
                         std(profits_std_true),  std(profits_std_saa)],
    First_Stage_Facs  = [string(opened_ddu), string(opened_ddu),
                         string(opened_std), string(opened_std)],
    Active_Region     = [region1_ddu, region1_ddu, region1_std, region1_std],
    Final_LB          = [result_ddu.lb,  result_ddu.lb,  lb_std_final, lb_std_final],
    Iterations        = [result_ddu.iters, result_ddu.iters,
                         result_std.iters, result_std.iters],
    Wall_Time_s       = [round(t_ddu; digits=1), round(t_ddu; digits=1),
                         round(t_std; digits=1), round(t_std; digits=1)],
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

n_ddu = length(ddu_lb_hist)
n_std = length(std_lb_hist)

conv_df = vcat(
    DataFrame(policy      = fill("DDU-SDDiP", n_ddu),
              iteration   = 1:n_ddu,
              time_s      = ddu_time_hist,
              lower_bound = ddu_lb_hist),
    DataFrame(policy      = fill("SDDiP", n_std),
              iteration   = 1:n_std,
              time_s      = std_time_hist,
              lower_bound = std_lb_hist),
)
CSV.write(joinpath(out_dir, "convergence_data_$(FILE_TAG).csv"), conv_df)

# ── Part 6: Opening statistics CSVs ──────────────────────────────────────────
# opening_stats_*.csv   — per-facility open frequency at each stage (long format)
# opening_summary_*.csv — per-stage aggregates: avg # open and modal pattern

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
    compute_opening_freq(x_hist_ddu_true, "DDU-SDDiP", "true"),
    compute_opening_freq(x_hist_ddu_saa,  "DDU-SDDiP", "SAA"),
    compute_opening_freq(x_hist_std_true, "SDDiP",     "true"),
    compute_opening_freq(x_hist_std_saa,  "SDDiP",     "SAA"),
)
open_summ_df = vcat(
    compute_opening_summary(x_hist_ddu_true, "DDU-SDDiP", "true"),
    compute_opening_summary(x_hist_ddu_saa,  "DDU-SDDiP", "SAA"),
    compute_opening_summary(x_hist_std_true, "SDDiP",     "true"),
    compute_opening_summary(x_hist_std_saa,  "SDDiP",     "SAA"),
)

CSV.write(joinpath(out_dir, "opening_stats_$(FILE_TAG).csv"),   open_freq_df)
CSV.write(joinpath(out_dir, "opening_summary_$(FILE_TAG).csv"), open_summ_df)

# ── Part 7: Convergence plots ─────────────────────────────────────────────────

gr()

p1 = plot(1:n_ddu, ddu_lb_hist;
    label     = "DDU-SDDiP",
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

p2 = plot(ddu_time_hist, ddu_lb_hist;
    label     = "DDU-SDDiP",
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
# 2×2 grid: rows = policy (DDU-SDDiP / SDDiP), columns = evaluation distribution

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

p_ddu_true = opening_heatmap(x_hist_ddu_true, "DDU-SDDiP / true dist")
p_ddu_saa  = opening_heatmap(x_hist_ddu_saa,  "DDU-SDDiP / SAA dist")
p_std_true = opening_heatmap(x_hist_std_true, "SDDiP / true dist")
p_std_saa  = opening_heatmap(x_hist_std_saa,  "SDDiP / SAA dist")

fig_open = plot(p_ddu_true, p_ddu_saa, p_std_true, p_std_saa;
    layout = (2, 2), size = (1200, 800), margin = 5Plots.mm)
savefig(fig_open, joinpath(out_dir, "opening_freq_plot_$(FILE_TAG).png"))

println("\nFiles written to: $out_dir")
println("  comparison_summary_$(FILE_TAG).csv")
println("  convergence_data_$(FILE_TAG).csv")
println("  opening_stats_$(FILE_TAG).csv")
println("  opening_summary_$(FILE_TAG).csv")
println("  convergence_plot_$(FILE_TAG).png")
println("  opening_freq_plot_$(FILE_TAG).png")
