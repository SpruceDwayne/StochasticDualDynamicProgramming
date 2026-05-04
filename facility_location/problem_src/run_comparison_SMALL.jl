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
using Distributions: Categorical

include(joinpath(@__DIR__, "distributions.jl"))
include(joinpath(@__DIR__, "subproblem_G.jl"))

# ── Instance ──────────────────────────────────────────────────────────────────
const INST_PATH = joinpath(@__DIR__, "..", "..", "instance_data_small_D.json")
const INST_NAME = replace(splitext(basename(INST_PATH))[1], "instance_data_" => "")
const inst = load_instance(INST_PATH)
@printf("Instance: %s | %d facilities | %d customers | %d zones | T=%d | max_demand=%d\n",
    inst.interaction_type, inst.num_facilities, inst.num_customers,
    inst.num_zones, inst.T, inst.max_demand)

const nI        = inst.num_facilities
const n_regions = length(inst.activation_regions)
const x0        = zeros(Float64, nI)
const ζ_init    = 1
const M_BIG     = 1e5

# ── Exact distribution enumeration ────────────────────────────────────────────
# For each region d, enumerate all (max_demand+1)^num_customers demand vectors
# and assign exact joint probabilities (customers are independent given region).
function generate_exact_scenarios(inst::InstanceData)
    nJ   = inst.num_customers
    vals = 0:inst.max_demand
    exact_Xi = Dict{Int, Vector{Any}}()
    exact_p  = Dict{Int, Vector{Float64}}()

    for d in 1:length(inst.activation_regions)
        zone_active = region_to_zone_active(inst, d)
        pmfs = compute_demand_distributions(inst, zone_active)  # pmfs[j][k+1] = P(ξ_j=k)

        Xi = Any[]
        p  = Float64[]
        for combo in Iterators.product(fill(vals, nJ)...)
            ξ    = collect(Int, combo)
            prob = prod(pmfs[j][ξ[j]+1] for j in 1:nJ)
            push!(Xi, ξ)
            push!(p, prob)
        end
        p ./= sum(p)  # normalise away any floating-point drift

        exact_Xi[d] = Xi
        exact_p[d]  = p
    end
    return exact_Xi, exact_p
end

const exact_Xi, exact_p = generate_exact_scenarios(inst)
n_exact = length(exact_Xi[1])
@printf("Exact scenarios per region: %d  (= %d^%d)\n",
    n_exact, inst.max_demand + 1, inst.num_customers)

const ddu_regions = DDURegion[
    DDURegion(d, exact_Xi[d], exact_p[d])
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

# ── Exogenous stage builder (Standard SDDiP baseline) ─────────────────────────
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

            # Demand allocation: only at non-root stages (stage 1 is investment only)
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
          (ζ) -> ddu_regions[ζ].Xi[rand(Categorical(ddu_regions[ζ].pXi))], unit_weight,
          (t, ζ) -> (ddu_regions[ζ].Xi, ddu_regions[ζ].pXi), noop_ctx, noop_key),
    Stage(3, nI, make_stage_builder(inst, 3, GRB_ENV),
          (ζ) -> ddu_regions[ζ].Xi[rand(Categorical(ddu_regions[ζ].pXi))], unit_weight,
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
    @printf("DDU iter %4d | new cuts: %2d | total: %3d | LB=%.4f | ΔLB=%.3e\n",
        it, stats.new_cuts, stats.total_cuts, stats.lb, stats.Δlb)
end

println("\n" * "="^72)
println("DDU-SDDiP  (exact distribution)")
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

ω1_ddu = m_ddu.stages[1].sampler(ζ_init)
model1_ddu, _, _, misc1_ddu = get_or_build_ddu_model!(m_ddu, 1, ω1_ddu, x0)
JuMP.optimize!(model1_ddu)
x1_ddu_vals = JuMP.value.(misc1_ddu[:x_next])
x1_ddu_int  = [v > 0.5 ? 1 : 0 for v in x1_ddu_vals]
opened_ddu  = findall(v -> v > 0.5, x1_ddu_vals)
region1_ddu = identify_region(inst, x1_ddu_int)

@printf("\nDDU done: %d iters | LB=%.4f | %.1f s\n", result_ddu.iters, result_ddu.lb, t_ddu)
@printf("  Facilities opened: %s | region d=%d\n", string(opened_ddu), region1_ddu)

# ── Part 2: Standard SDDiP (ignoring DDU) ────────────────────────────────────

rng_std = MersenneTwister(456)

# Exact scenarios for region 1 used throughout (exogenous baseline)
const std_Xi1 = exact_Xi[1]
const std_p1  = exact_p[1]

stages_std = Stage[
    Stage(1, nI, make_exogenous_stage_builder(inst, 1, GRB_ENV),
          ()  -> nothing, unit_weight,
          (t, ctx) -> (std_Xi1, std_p1),
          (t, ctx, ω) -> nothing, x -> nothing),
    Stage(2, nI, make_exogenous_stage_builder(inst, 2, GRB_ENV),
          ()  -> std_Xi1[rand(rng_std, Categorical(std_p1))], unit_weight,
          (t, ctx) -> (std_Xi1, std_p1),
          (t, ctx, ω) -> nothing, x -> nothing),
    Stage(3, nI, make_exogenous_stage_builder(inst, 3, GRB_ENV),
          ()  -> std_Xi1[rand(rng_std, Categorical(std_p1))], unit_weight,
          (t, ctx) -> (std_Xi1, std_p1),
          (t, ctx, ω) -> nothing, x -> nothing),
]

m_std = SDDP(stages_std)

std_lb_hist   = Float64[]
std_time_hist = Float64[]
t_std_start   = time()
ω1_std        = expected_d1

function std_logfn(it, stats)
    lb = compute_std_lb!(m_std, x0, ω1_std, inst)
    push!(std_lb_hist,   lb)
    push!(std_time_hist, time() - t_std_start)
    @printf("STD iter %4d | new cuts: %2d | total: %3d | LB=%.4f | ΔV=%.3e\n",
        it, stats.new_cuts, stats.total_cuts, lb, stats.ΔV)
end

println("\n" * "="^72)
println("Standard SDDiP  (ignoring DDU — region 1 demand throughout)")
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

vf2_std     = m_std.V[2]
model1_std, _, _, misc1_std = SDDPBAPE.get_or_build_model!(m_std, 1, vf2_std, ω1_std, nothing, x0)
JuMP.optimize!(model1_std)
x1_std_vals = JuMP.value.(misc1_std[:x_next])
x1_std_int  = [v > 0.5 ? 1 : 0 for v in x1_std_vals]
opened_std  = findall(v -> v > 0.5, x1_std_vals)
region1_std = identify_region(inst, x1_std_int)
lb_std_final = JuMP.objective_value(model1_std)

@printf("\nSTD done: %d iters | LB=%.4f | %.1f s\n", result_std.iters, lb_std_final, t_std)
@printf("  Facilities opened: %s | region d=%d\n", string(opened_std), region1_std)

# ── Part 3: Out-of-sample simulation (TRUE DDU demand) ────────────────────────

const N_SIM = 1000

function simulate_policy(m, is_ddu::Bool; n_paths::Int = N_SIM, seed::Int = 1111)
    rng     = MersenneTwister(seed)
    profits = zeros(Float64, n_paths)

    for s in 1:n_paths
        x_prev = copy(x0)
        total  = 0.0

        for t in 1:inst.T
            x_int  = [v > 0.5 ? 1 : 0 for v in x_prev]
            d_true = identify_region(inst, x_int)
            za     = region_to_zone_active(inst, d_true)
            # Stage 1 is a pure investment stage with no demand realization.
            ξ      = t == 1 ? nothing :
                     Float64[rand(rng, _customer_dist(inst, j, za))
                              for j in 1:inst.num_customers]

            if is_ddu
                model, _, θ_var, misc = get_or_build_ddu_model!(m, t, ξ, x_prev)
            else
                vf_next = t < inst.T ? m.V[t + 1] : ValueFn{Float64}()
                model, _, θ_var, misc = SDDPBAPE.get_or_build_model!(m, t, vf_next, ξ, nothing, x_prev)
            end
            JuMP.optimize!(model)

            obj_val    = JuMP.objective_value(model)
            θ_val      = JuMP.value(θ_var)
            stage_prof = -(obj_val - θ_val)
            total     += stage_prof

            x_prev = Float64[JuMP.value(v) > 0.5 ? 1.0 : 0.0
                              for v in misc[:x_next]]
        end

        profits[s] = total
    end

    return profits
end

println("\nSimulating DDU policy  ($N_SIM paths) ...")
profits_ddu = simulate_policy(m_ddu, true)
println("Simulating STD policy  ($N_SIM paths) ...")
profits_std = simulate_policy(m_std, false)

# ── Part 4: Summary ───────────────────────────────────────────────────────────

summary_df = DataFrame(
    Instance_Type    = [INST_NAME,           INST_NAME],
    Exact_Scenarios  = [n_exact,             n_exact],
    Policy           = ["DDU-SDDiP",         "Standard SDDiP"],
    Avg_Profit       = [mean(profits_ddu),    mean(profits_std)],
    Std_Dev          = [std(profits_ddu),     std(profits_std)],
    First_Stage_Facs = [string(opened_ddu),   string(opened_std)],
    Active_Region    = [region1_ddu,          region1_std],
    Final_LB         = [result_ddu.lb,        lb_std_final],
    Iterations       = [result_ddu.iters,     result_std.iters],
    Wall_Time_s      = [round(t_ddu; digits=1), round(t_std; digits=1)],
)

out_dir = joinpath(@__DIR__, "..", "results", INST_NAME)
mkpath(out_dir)
const FILE_TAG = uppercase(INST_NAME) * "_exact"

CSV.write(joinpath(out_dir, "comparison_summary_$(FILE_TAG).csv"), summary_df)

println("\n" * "="^72)
println("SUMMARY")
println("="^72)
for row in eachrow(summary_df)
    @printf("%-18s | Avg profit: %9.2f | Std: %7.2f | Facs: %-12s | Region: %d\n",
        row.Policy, row.Avg_Profit, row.Std_Dev, row.First_Stage_Facs, row.Active_Region)
end
println("="^72)

# ── Part 5: Convergence data and plots ────────────────────────────────────────

n_ddu = length(ddu_lb_hist)
n_std = length(std_lb_hist)

conv_df = vcat(
    DataFrame(policy      = fill("DDU-SDDiP", n_ddu),
              iteration   = 1:n_ddu,
              time_s      = ddu_time_hist,
              lower_bound = ddu_lb_hist),
    DataFrame(policy      = fill("Standard SDDiP", n_std),
              iteration   = 1:n_std,
              time_s      = std_time_hist,
              lower_bound = std_lb_hist),
)
CSV.write(joinpath(out_dir, "convergence_data_$(FILE_TAG).csv"), conv_df)

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
    label     = "Standard SDDiP",
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
    label     = "Standard SDDiP",
    linewidth = 2,
    color     = :red,
    linestyle = :dash,
)

fig = plot(p1, p2; layout = (1, 2), size = (1200, 450), margin = 5Plots.mm)
savefig(fig, joinpath(out_dir, "convergence_plot_$(FILE_TAG).png"))

println("\nFiles written to: $out_dir")
println("  comparison_summary_$(FILE_TAG).csv")
println("  convergence_data_$(FILE_TAG).csv")
println("  convergence_plot_$(FILE_TAG).png")
