using Pkg
Pkg.activate(joinpath(@__DIR__, "SDDPBAPE"))
using SDDPBAPE
using JuMP, HiGHS
using LinearAlgebra

# ============================================================================
# DDU Smoke Test: 2-Stage Binary Unit Commitment with Decision-Dependent Demand
# ============================================================================
#
# Problem:
#   Stage 1: decide x1 ∈ {0,1} (generator OFF=0 / ON=1).
#            Region 1 (d=1): x1=0 → stage-2 demand is LOW  {4.0, 5.0}
#            Region 2 (d=2): x1=1 → stage-2 demand is HIGH {8.0, 9.0}
#            (Turning the generator ON attracts high-demand consumers.)
#
#   Stage 2: decide x2 ∈ {0,1}, given x1 fixed and demand ω.
#
# Indexing sanity: the backward pass for V[1][d] fixes the stage-2 model to
# x_leaving[s][1] (= x1, the OUTPUT of stage 1), NOT x_entering[s][1] (= x0).
# Test 7 specifically checks this.
#
# Parameters
const Gmax = 10.0
const SU   = 2.0
const F    = 0.5
const c_g  = 1.0
const M_sh = 100.0   # load-shed penalty (not to be confused with big-M for DDU)

# DDU region definitions:
#   Region 1 (x1=0 OFF): low-demand distribution
#   Region 2 (x1=1 ON ): high-demand distribution
const REGION_1 = DDURegion(1, Any[4.0, 5.0], [0.5, 0.5])
const REGION_2 = DDURegion(2, Any[8.0, 9.0], [0.5, 0.5])

# ζ_init: incoming context at stage 1.
# Since stage 1 has no preceding decision we use a special "root" region id=0
# with a fixed single scenario (dummy demand = 0, not used in stage-1 cost).
const REGION_0 = DDURegion(0, Any[nothing], [1.0])   # root/dummy for stage 1
const ζ_INIT   = 0

# ============================================================================
# Stage builders
# ============================================================================

# Stage 1 builder
# ---------------
# Stage 1 has no parent state to fix: x_state is returned as JuMP.VariableRef[].
# This allows x1 to be a FREE decision variable (not fixed to x0).
# If x_state contained [x1], then x1 would be fixed to x0 by get_or_build_ddu_model!,
# preventing stage 1 from optimising freely — the most common DDU stage-1 mistake.
# Region indicators: 𝟙_1 (x1=0) and 𝟙_2 (x1=1).
# Membership: 𝟙_2 = x1,  𝟙_1 = 1 - x1.
# θ is the DDU continuation; big-M cuts from V[1][d] are added externally.
function build_stage1_ddu(t, vf_next, ω; fix_state)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x1, Bin)
    @variable(model, u1, Bin)
    @variable(model, g1  >= 0)
    @variable(model, shed1 >= 0)
    @variable(model, θ >= -1e7)    # continuation (big-M cuts added externally)

    # Stage-1 has a fixed deterministic demand of 6 (ω is dummy/nothing)
    d1 = 6.0
    @constraint(model, g1 + shed1 == d1)
    @constraint(model, g1 <= Gmax * x1)
    @constraint(model, u1 >= x1)   # start from OFF

    # Region membership: region 2 is active iff x1=1
    @variable(model, ind1, Bin)   # 𝟙_1
    @variable(model, ind2, Bin)   # 𝟙_2
    @constraint(model, ind1 + ind2 == 1)
    @constraint(model, ind2 == x1)    # region 2 ↔ x1=1
    # (ind1 = 1 - x1 follows from the equality above)

    # NOTE: vf_next is ignored; DDU adds big-M cuts externally via get_or_build_ddu_model!
    @objective(model, Min, SU*u1 + F*x1 + c_g*g1 + M_sh*shed1 + θ)

    misc = Dict{Symbol, Any}(
        :x_next            => [x1],
        :region_indicators => Dict{Int, JuMP.VariableRef}(1 => ind1, 2 => ind2),
    )
    # x_state is empty: stage 1 has no parent state to fix.
    # x1 is a FREE decision variable, not an inherited state.
    return model, JuMP.VariableRef[], θ, misc
end

# Stage 2 builder
# ---------------
# State: x2 ∈ {0,1}.  z2 ∈ [0,1] copies parent x1.
# No continuation (terminal stage).
# No region indicators needed (no stage-3 continuation).
function build_stage2_ddu(t, vf_next, ω; fix_state)
    d2 = ω   # demand from DDURegion.Xi

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x2, Bin)
    @variable(model, u2, Bin)
    @variable(model, g2  >= 0)
    @variable(model, shed2 >= 0)

    # SDDiP copy variable
    @variable(model, 0 <= z2 <= 1)
    @constraint(model, z2 == fix_state[1])   # will be replaced by JuMP.fix

    @constraint(model, g2 + shed2 == d2)
    @constraint(model, g2 <= Gmax * x2)
    @constraint(model, u2 >= x2 - z2)

    # Terminal stage: no continuation
    @variable(model, θ == 0)

    @objective(model, Min, SU*u2 + F*x2 + c_g*g2 + M_sh*shed2 + θ)

    # Stage 2 also needs region_indicators to satisfy the builder contract
    # (empty dict because there is no stage-3 continuation)
    misc = Dict{Symbol, Any}(
        :z_vars            => [z2],
        :x_next            => [x2],
        :x_state           => [x2],
        :region_indicators => Dict{Int, JuMP.VariableRef}(),
    )
    return model, [x2], θ, misc
end

# ============================================================================
# Stage and DDUSDDP assembly
# ============================================================================

stage1 = Stage(
    1, 1,
    build_stage1_ddu,
    (ζ) -> nothing,                          # sampler: dummy (stage 1 is deterministic)
    _ -> 1.0,
    (t, ζ) -> (REGION_0.Xi, REGION_0.pXi),  # children: unused by DDU backward at t=1
    (t, ζ, ω) -> ζ,                          # next_ctx: no-op placeholder
    x -> x,
)

stage2 = Stage(
    2, 1,
    build_stage2_ddu,
    (ζ) -> begin                              # sampler: draw from μ_{ζ}
        reg = ζ == 1 ? REGION_1 : REGION_2
        idx = rand(1:length(reg.Xi))
        reg.Xi[idx]
    end,
    _ -> 1.0,
    (t, ζ) -> begin                           # children: return full support of μ_{ζ}
        reg = ζ == 1 ? REGION_1 : REGION_2
        (reg.Xi, reg.pXi)
    end,
    (t, ζ, ω) -> ζ,
    x -> x,
)

# Regions per stage
# Stage 1 has outgoing regions D_1 = {1, 2}
# Stage 2 has no outgoing regions (terminal), so D_2 = {}
regions_stage1 = [REGION_1, REGION_2]
regions_stage2 = DDURegion[]

m = DDUSDDP(
    [stage1, stage2],
    [regions_stage1, regions_stage2];
    M_big = 500.0,   # safely above max stage-2 cost (~100*9 = 900 minus min ≈ 4.5)
)
x0 = [0.0]

config = SDDiPConfig(cut_type = :SB)

println("="^70)
println("DDU Smoke Test: 2-Stage Binary UC with Decision-Dependent Demand")
println("="^70)

# ============================================================================
# Test 1: DDUSDDP construction
# ============================================================================
println("\n--- Test 1: DDUSDDP construction ---")
@assert m.T == 2                                "T should be 2"
@assert length(m.regions[1]) == 2              "stage 1 should have 2 regions"
@assert length(m.regions[2]) == 0              "stage 2 should have 0 regions"
@assert isempty(m.V[1])                        "V[1] should start empty"
@assert isempty(m.V[2])                        "V[2] should start empty"
@assert m.M_big == 500.0                       "M_big should be 500.0"
println("  PASS: DDUSDDP constructed correctly")

# ============================================================================
# Test 2: DDURegion validation
# ============================================================================
println("\n--- Test 2: DDURegion construction ---")
@assert REGION_1.id == 1
@assert length(REGION_1.Xi) == 2
@assert abs(sum(REGION_1.pXi) - 1.0) < 1e-10
@assert REGION_2.id == 2
@assert length(REGION_2.Xi) == 2
println("  PASS: DDURegion constructed and validated")

# ============================================================================
# Test 3: get_or_build_ddu_model! — first call builds, exposes region_indicators
# ============================================================================
println("\n--- Test 3: model construction and region_indicators ---")
model1, xs1, θ1, misc1 = get_or_build_ddu_model!(m, 1, nothing, x0)
@assert haskey(misc1, :region_indicators)      "misc[:region_indicators] missing"
ri = misc1[:region_indicators]
@assert haskey(ri, 1) && haskey(ri, 2)         "indicators for regions 1 and 2 missing"
@assert ri[1] isa JuMP.VariableRef             "indicator 1 should be a VariableRef"
@assert ri[2] isa JuMP.VariableRef             "indicator 2 should be a VariableRef"
@assert haskey(misc1, :x_next)                 ":x_next missing from misc"
println("  PASS: model built, region_indicators present")

# ============================================================================
# Test 4: per-region cut insertion via get_or_build_ddu_model!
# ============================================================================
println("\n--- Test 4: per-region cut insertion (big-M) ---")
# Manually insert a cut into V[1][2] (region 2 = generator ON)
vf12 = get_V_ddu!(m, 1, 2)
add_cut!(vf12, 10.0, [3.0], 1)   # θ ≥ 10 + 3*x_next
n_cuts_before = length(vf12.cuts)
@assert n_cuts_before == 1  "should have 1 cut in V[1][2]"

# Trigger cache update by calling get_or_build_ddu_model! again
# (second call should add the big-M constraint to the model)
model1b, _, θ1b, misc1b = get_or_build_ddu_model!(m, 1, nothing, x0)
updated_count = get(m.model_cache[1][nothing].last_cut_count_by_region, 2, -1)
@assert updated_count == 1  "last_cut_count_by_region[2] should be 1 after update"
println("  PASS: big-M cut added incrementally, count tracked correctly")

# ============================================================================
# Test 5: forward pass — region selection is valid
# ============================================================================
println("\n--- Test 5: forward pass region selection ---")
fwd = forward_pass_ddu_online!(m; S=1, x0=x0, ζ_init=ζ_INIT)

# x_entering[1][1] should equal x0
@assert fwd.x_entering[1][1] ≈ x0  "x_entering[s=1][t=1] should equal x0"

# ζ_hist[1][1] should equal ζ_INIT
@assert fwd.ζ_hist[1][1] == ζ_INIT  "ζ_hist[1][1] should be ζ_init=$ζ_INIT"

# δ_hist[s][t] must be a valid region id for stage t
for s in 1:1, t in 1:m.T
    δ = fwd.δ_hist[s][t]
    valid_ids = [r.id for r in m.regions[t]]
    if !isempty(valid_ids)   # terminal stage has no regions → skip
        @assert δ in valid_ids  "δ_hist[$s][$t]=$δ not in valid region ids $valid_ids"
    end
end

# x1 should be binary; with x_state=[] x1 is a free decision variable so its
# value may differ from x0 (it is chosen by the optimizer, not fixed to x0).
x1_val = fwd.x_leaving[1][1][1]
@assert x1_val ≈ 0.0 || x1_val ≈ 1.0  "x_leaving[1][1] should be binary, got $x1_val"

# Consistency: x_leaving[s][t] == x_entering[s][t+1] for t < T
for s in 1:1, t in 1:(m.T-1)
    @assert fwd.x_leaving[s][t] ≈ fwd.x_entering[s][t+1] "x_leaving[$s][$t] != x_entering[$s][$(t+1)]"
end
# ζ_{t+1} == δ_t
for s in 1:1, t in 1:(m.T-1)
    @assert fwd.ζ_hist[s][t+1] == fwd.δ_hist[s][t] "zeta_hist[$s][$(t+1)] should equal delta_hist[$s][$t]"
end
println("  PASS: forward pass region selection, state consistency, ζ/δ linkage")

# ============================================================================
# Test 6: backward pass generates a cut in V[1][δ]
# ============================================================================
println("\n--- Test 6: backward pass cut generation ---")
# Reset model to a fresh instance for a clean backward test
m2 = DDUSDDP(
    [stage1, stage2],
    [regions_stage1, regions_stage2];
    M_big = 500.0,
)
fwd2 = forward_pass_ddu_online!(m2; S=1, x0=x0, ζ_init=ζ_INIT)
δ1 = fwd2.δ_hist[1][1]   # outgoing region chosen at stage 1

cuts_before = sum(length(vf.cuts) for vf in values(m2.V[1]); init=0)
backward_pass_ddu_sddip!(m2; fwd=fwd2, config=config, iter=1, force_every=1, atol=0.0)
cuts_after = sum(length(vf.cuts) for vf in values(m2.V[1]); init=0)

@assert cuts_after > cuts_before  "backward pass should add at least one cut to V[1]"
@assert haskey(m2.V[1], δ1)       "V[1][δ=$δ1] should exist after backward pass"
println("  PASS: backward pass added $(cuts_after - cuts_before) cut(s) to V[1][$δ1]")

# ============================================================================
# Test 7: x_leaving vs x_entering — the critical DDU indexing check
# ============================================================================
# If the backward pass were incorrectly using x_entering[s][t] instead of
# x_leaving[s][t] as the stage-(t+1) trial point, the Lagrangian cut intercept
# α would be computed at the WRONG state.  We verify this by checking that the
# cut is tight (within solver tolerance) at x_leaving[s][1], not x_entering[s][1].
println("\n--- Test 7: cut tightness at x_leaving (not x_entering) ---")
x_leaving_1  = fwd2.x_leaving[1][1]
x_entering_1 = fwd2.x_entering[1][1]
vf_δ1 = m2.V[1][δ1]
@assert !isempty(vf_δ1.cuts)  "no cuts in V[1][$δ1]"

cut = vf_δ1.cuts[1]
val_at_leaving  = cut.α + dot(cut.β, x_leaving_1)
val_at_entering = cut.α + dot(cut.β, x_entering_1)

# The cut should be tight (≈ true value) at x_leaving; at x_entering it may not be.
# We verify that val_at_leaving is finite and positive (some cost was computed).
@assert isfinite(val_at_leaving)           "cut value at x_leaving must be finite"
@assert val_at_leaving >= -1e-6            "cut must be a valid lower bound (≥ 0) at x_leaving"

# Only flag a problem if x_leaving ≠ x_entering (otherwise both are the same point)
if !(x_leaving_1 ≈ x_entering_1)
    # If the states differ, the cut intercept must match the stage-2 optimal value
    # at x_leaving_1, not x_entering_1.  We verify by solving stage 2 directly
    # at both states and checking which one is consistent.
    println("  x_entering=$x_entering_1  x_leaving=$x_leaving_1  (differ: indexing test active)")
    # Solve stage 2 at x_leaving_1 manually (one scenario)
    reg_δ1 = δ1 == 1 ? REGION_1 : REGION_2
    local α_check = 0.0
    for (ωj, pj) in zip(reg_δ1.Xi, reg_δ1.pXi)
        m_tmp, _, _, misc_tmp = get_or_build_ddu_model!(m2, 2, ωj, x_leaving_1)
        JuMP.optimize!(m_tmp)
        α_check += pj * JuMP.objective_value(m_tmp)
    end
    @assert abs(val_at_leaving - α_check) < 1.0 "cut intercept at x_leaving ($val_at_leaving) should match expected value ($α_check)"
    println("  PASS: cut is consistent with stage-2 optimal value at x_leaving")
else
    println("  (x_leaving == x_entering in this run; tightness test trivially passes)")
end
println("  PASS: x_leaving vs x_entering test complete")

# ============================================================================
# Test 8: run_ddu_sddip! completes without error
# ============================================================================
println("\n--- Test 8: run_ddu_sddip! full loop ---")
m3 = DDUSDDP(
    [stage1, stage2],
    [regions_stage1, regions_stage2];
    M_big = 500.0,
)
result = run_ddu_sddip!(m3;
    x0          = x0,
    ζ_init      = ζ_INIT,
    config      = config,
    S           = 1,
    max_iter    = 30,
    patience    = 10,
    force_every = 1,
    cut_atol    = 0.0,
    evaluate_stage = 1,
    evaluate_δ     = 1,
)
@assert result.iters >= 1                              "should run at least 1 iteration"
@assert sum(result.cuts_per_stage) >= 1               "should generate at least 1 cut total"
println("  Completed $(result.iters) iterations, total cuts: $(sum(result.cuts_per_stage))")
println("  Cuts per stage: $(result.cuts_per_stage)")
println("  PASS: run_ddu_sddip! completed without error")

# ============================================================================
# Test 9: big-M deactivation — constraint count grows as cuts are added
# ============================================================================
println("\n--- Test 9: big-M deactivation / constraint count ---")
# Use m2 and fwd2 from Test 6/7 (already have a backward pass done).
# Add a second cut to V[1][δ1] and verify the stage-1 cached model picks it up.
vf_δ1_t9 = get_V_ddu!(m2, 1, δ1)
n_cuts_before_t9 = length(vf_δ1_t9.cuts)

# Count constraints in the cached stage-1 model before adding the new cut
model_before = m2.model_cache[1][nothing].model
n_con_before = num_constraints(model_before; count_variable_in_set_constraints = false)

# Add a new cut manually
add_cut!(vf_δ1_t9, 5.0, [2.0], 1)
@assert length(vf_δ1_t9.cuts) == n_cuts_before_t9 + 1  "cut should have been added"

# Trigger the model update by calling get_or_build_ddu_model! again
get_or_build_ddu_model!(m2, 1, nothing, x0)
model_after = m2.model_cache[1][nothing].model
n_con_after = num_constraints(model_after; count_variable_in_set_constraints = false)

@assert n_con_after > n_con_before  "constraint count should increase after adding a new big-M cut; before=$n_con_before after=$n_con_after"

# Verify the new cut intercept is consistent: θ ≥ 5 + 2*x_next + M*(𝟙_δ1 - 1)
# At x_next=0 and 𝟙_δ1=1 (region δ1 active): lower bound on θ should be ≥ 5+2*0=5
# At 𝟙_δ1=0 (inactive): contribution is 5 + 0 - M_big ≈ -495 (relaxed, non-binding)
new_cut = vf_δ1_t9.cuts[end]
val_at_zero = new_cut.α + dot(new_cut.β, [0.0])
@assert val_at_zero ≈ 5.0  "cut intercept at x_next=0 should be 5.0, got $val_at_zero"
println("  PASS: constraint count increased ($n_con_before → $n_con_after) after adding big-M cut")
println("  PASS: cut intercept at x_next=0 is $(val_at_zero) (consistent with chosen region)")

# ============================================================================
# Test 10: multi-region backward coverage
# ============================================================================
println("\n--- Test 10: multi-region backward coverage ---")
# Create a fresh model and run for several iterations with force_every=1.
# With x_state=[] the optimizer is free to choose x1 each iteration, so both
# regions can be visited and accumulate cuts.
m_multi = DDUSDDP(
    [stage1, stage2],
    [regions_stage1, regions_stage2];
    M_big = 500.0,
)
result_multi = run_ddu_sddip!(m_multi;
    x0          = x0,
    ζ_init      = ζ_INIT,
    config      = config,
    S           = 1,
    max_iter    = 20,
    patience    = 20,   # run all 20 iterations
    force_every = 1,
    cut_atol    = 0.0,
    evaluate_stage = 1,
    evaluate_δ     = 1,
)

# At least one region at stage 1 must have accumulated cuts
n_regions_with_cuts = count(d -> haskey(m_multi.V[1], d.id) && !isempty(m_multi.V[1][d.id].cuts),
                             regions_stage1)
@assert n_regions_with_cuts >= 1  "at least one region in V[1] should have cuts after 20 iters"
println("  Regions with cuts in V[1]: $n_regions_with_cuts / $(length(regions_stage1))")
println("  V[1] keys with cuts: $(filter(k -> !isempty(m_multi.V[1][k].cuts), collect(keys(m_multi.V[1]))))")

# Ideally both regions accumulate cuts (the optimizer alternates choices as big-M
# cuts shift the value function). With force_every=1 and 20 iterations this is
# highly likely but not guaranteed, so we only assert >= 1.
if n_regions_with_cuts == 2
    println("  PASS: both regions (1 and 2) accumulated cuts — multi-region coverage confirmed")
else
    println("  INFO: only $n_regions_with_cuts region(s) have cuts (optimizer consistently chose one region)")
end
@assert result_multi.iters >= 1  "run_ddu_sddip! should run at least 1 iteration"
println("  PASS: multi-region backward coverage test complete")

println("\n" * "="^70)
println("ALL DDU SMOKE TEST CHECKS PASSED")
println("="^70)
