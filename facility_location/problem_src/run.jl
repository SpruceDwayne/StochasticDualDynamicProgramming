import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "SDDPBAPE"))
Pkg.instantiate()   # resolve + install any packages in Project.toml not yet in Manifest

using SDDPBAPE
using JuMP, HiGHS
using Printf

include(joinpath(@__DIR__, "distributions.jl"))   # pulls in data.jl + Distributions
include(joinpath(@__DIR__, "subproblem.jl"))

# ── Load instance ─────────────────────────────────────────────────────────────
const INST_PATH = joinpath(@__DIR__, "..", "..", "instance_data.json")
const inst = load_instance(INST_PATH)
@printf("Instance: %d facilities | %d customers | %d zones | T=%d\n",
    inst.num_facilities, inst.num_customers, inst.num_zones, inst.T)
# ── Generate fixed SAA scenario sets (ONCE before the algorithm) ──────────────
# scenarios[d] is an (N × num_customers) integer matrix; each row is one demand
# realisation drawn i.i.d. per customer from the BetaBinomial PMF for region d.
const N_SAA = 50    
const saa   = generate_saa_scenarios(inst; N = N_SAA, seed = 123)
@printf("SAA scenarios: %d regions × %d scenarios\n", length(saa), N_SAA)

# ── Build DDURegion objects ───────────────────────────────────────────────────
# DDURegion(id, Xi, pXi) where Xi[n] is the n-th demand vector and pXi is uniform.
const n_regions  = length(inst.activation_regions)   # 8
const uniform_p  = fill(1.0 / N_SAA, N_SAA)

const ddu_regions = DDURegion[
    DDURegion(d, Any[saa[d][n, :] for n in 1:N_SAA], uniform_p)
    for d in 1:n_regions
]

# ── Stage-1 demand: expected demand under region 1 ────────────────────────────
# x_0 = 0  →  no zones active  →  region 1.
# Using expected (not sampled) demand for stage 1 makes compute_ddu_lb! call the
# same model every iteration, giving a deterministic, monotone lower-bound sequence.
const zone_active_0 = region_to_zone_active(inst, 1)   # all false for region 1
const pmfs_region1  = compute_demand_distributions(inst, zone_active_0)
const expected_d1   = Float64[
    sum(Float64(k) * pmfs_region1[j][k+1] for k in 0:inst.max_demand)
    for j in 1:inst.num_customers
]

# ── Samplers and children ─────────────────────────────────────────────────────
# stage-1 sampler: deterministic — always returns expected_d1 (ignores ζ)
stage1_sampler  = (ζ) -> expected_d1
# stages 2–T sampler: uniform sample from the N scenarios for region ζ
generic_sampler = (ζ) -> ddu_regions[ζ].Xi[rand(1:N_SAA)]
# children used by the backward pass: all N scenarios for region ζ with prob 1/N
children_fn     = (t, ζ) -> (ddu_regions[ζ].Xi, ddu_regions[ζ].pXi)
# context transition: region id is already set by the forward pass from δ_hist
noop_ctx        = (t, ζ, ω) -> ζ
noop_key        = x -> x
unit_weight     = _  -> 1.0

# ── Assemble Stage objects ────────────────────────────────────────────────────
const nI = inst.num_facilities

stages = Stage[
    Stage(1, nI, make_stage_builder(inst, 1),
          stage1_sampler,  unit_weight, children_fn, noop_ctx, noop_key),
    Stage(2, nI, make_stage_builder(inst, 2),
          generic_sampler, unit_weight, children_fn, noop_ctx, noop_key),
    Stage(3, nI, make_stage_builder(inst, 3),
          generic_sampler, unit_weight, children_fn, noop_ctx, noop_key),
    Stage(4, nI, make_stage_builder(inst, 4),
          generic_sampler, unit_weight, children_fn, noop_ctx, noop_key),
]

# ── Outgoing regions per stage ────────────────────────────────────────────────
# Stages 1–(T-1): 8 possible outgoing regions determined by x_t.
# Stage T (terminal): no outgoing regions (θ = 0, no continuation).
regions_per_stage = Vector{Vector{DDURegion}}(undef, inst.T)
for t in 1:inst.T - 1
    regions_per_stage[t] = ddu_regions
end
regions_per_stage[inst.T] = DDURegion[]

# ── Assemble DDUSDDP ──────────────────────────────────────────────────────────
# M_big: must dominate any cut value |α + β'x| for x ∈ {0,1}^8.
# Max stage profit ≈ 394 × 120 ≈ 47 280.  Use 1e5 for a comfortable margin.
const M_BIG = 1e5

m = DDUSDDP(stages, regions_per_stage; M_big = M_BIG, discount = 1.0)
@printf("DDUSDDP: T=%d | M_big=%.0e | %d outgoing regions per stage\n",m.T,M_BIG, n_regions)

# ── SDDiP configuration ───────────────────────────────────────────────────────
# Burn-in phase uses cheap IO cuts; main phase uses Lagrangian (tighter bounds).
config = SDDiPConfig(
    cut_type        = :lagrangian,
    burnin_iters    = 0,
    burnin_cut_type = :IO,
    level_cfg       = LevelMethodConfig(optimizer = HiGHS.Optimizer),
)

# ── Initial state ─────────────────────────────────────────────────────────────
const x0     = zeros(Float64, nI)   # no facilities open initially
const ζ_init = 1                    # region 1: x_0=0 → no zones active

println()
println("="^72)
@printf("DDU-SDDiP | T=%d | N_SAA=%d | M_big=%.0e | max_iter=200\n",inst.T, N_SAA, M_BIG)
println("="^72)

# ── Run the algorithm ─────────────────────────────────────────────────────────
# The training loop prints per-iteration stats by default (logfn=nothing):
#   iter  NNN | new cuts: NN | total: NNN | LB=X.XXXXXX | ΔLB=X.XXXe-XX
t_start = time()

result = run_ddu_sddip!(m;
    x0          = x0,
    ζ_init      = ζ_init,
    config      = config,
    S           = 1,          # forward samples per iteration
    max_iter    = 200,
    patience    = 30,         # stop if no new cuts for 30 consecutive iters
    force_every = 2,          # force a cut every 2 iters even if not improving
    cut_atol    = 1e-6,
)

elapsed = time() - t_start

# ── Print summary ─────────────────────────────────────────────────────────────
println()
println("="^72)
@printf("Done: %d iterations | Final LB = %.4f | Wall time: %.1f s\n", result.iters, result.lb, elapsed)

# Solve the stage-1 MILP one final time with all accumulated cuts to read the
# optimal first-stage decision.  The stage-1 model is cached and is already
# up to date; this just re-optimises it.
ω1_lb           = m.stages[1].sampler(ζ_init)            # = expected_d1 (fixed)
model1, _, _, misc1 = get_or_build_ddu_model!(m, 1, ω1_lb, x0)
JuMP.optimize!(model1)

x1_vals = JuMP.value.(misc1[:x_next])
opened  = findall(v -> v > 0.5, x1_vals)
x1_int  = [v > 0.5 ? 1 : 0 for v in x1_vals]
region1 = identify_region(inst, x1_int)

println()
println("First-stage decision:")
@printf("  Facilities opened: %s  (%d of k=%d budget used)\n",string(opened), length(opened), inst.k)
@printf("  Active region after stage 1: d=%d  (active zones: %s)\n",region1, string(sort(collect(inst.activation_regions[region1]))))

println()
@printf("Cuts per stage: %s\n", string(result.cuts_per_stage))
println("="^72)
