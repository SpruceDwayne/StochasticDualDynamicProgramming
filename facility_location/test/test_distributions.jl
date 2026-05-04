include(joinpath(@__DIR__, "..", "problem_src", "distributions.jl"))

const INSTANCE_PATH = joinpath(@__DIR__, "..", "..", "instance_data.json")
inst = load_instance(INSTANCE_PATH)

n_regions = length(inst.activation_regions)

# ── Test 1: x=zeros → all zones inactive → region 1, customer 1 mean = base mean ──
x0 = zeros(Int, inst.num_facilities)
b0 = zeros(Int, inst.num_zones)

d1 = identify_region(inst, x0, b0)
@assert d1 == 1 "Expected region 1 for all-inactive, got $d1"

zone_active_1 = region_to_zone_active(inst, d1)
pmfs_1 = compute_demand_distributions(inst, zone_active_1)
mean_cust1 = sum(k * pmfs_1[1][k+1] for k in 0:inst.max_demand)
expected_base = inst.max_demand * inst.alpha[1] / (inst.alpha[1] + inst.beta[1])
@assert isapprox(mean_cust1, expected_base; atol=1e-4) "Customer 1 mean mismatch: $mean_cust1 vs $expected_base"
println("Test 1 passed: region=$d1, customer 1 mean=$(round(mean_cust1; digits=4))")

# ── Test 2: x=ones → all zones active → last region ──
x1 = ones(Int, inst.num_facilities)
b1 = ones(Int, inst.num_zones)

d_all = identify_region(inst, x1, b1)
@assert d_all == n_regions "Expected region $n_regions for all-active, got $d_all"
println("Test 2 passed: all facilities open → region=$d_all (last region)")

# ── Test 3: PMFs are valid probability vectors ──
zone_active_all = region_to_zone_active(inst, d_all)
pmfs_all = compute_demand_distributions(inst, zone_active_all)
for j in 1:inst.num_customers
    s = sum(pmfs_all[j])
    @assert isapprox(s, 1.0; atol=1e-6) "PMF for customer $j does not sum to 1: $s"
    @assert all(pmfs_all[j] .>= 0) "PMF for customer $j has negative probabilities"
end
println("Test 3 passed: all PMFs are valid probability vectors")

# ── Test 4: sample_scenarios returns correct shapes and valid demands ──
scenarios, probs = sample_scenarios(inst, 1; N=50)
@assert size(scenarios) == (50, inst.num_customers) "scenarios shape mismatch"
@assert length(probs) == 50 && isapprox(sum(probs), 1.0; atol=1e-10) "probs invalid"
@assert all(0 .<= scenarios .<= inst.max_demand) "demand values out of range"
println("Test 4 passed: sample_scenarios shape and range OK")

# ── Test 5: opening facilities changes demand relative to no facilities ──
# (at least one customer's expected demand should differ)
any_change = false
for j in 1:inst.num_customers
    mean_none = sum(k * pmfs_1[j][k+1] for k in 0:inst.max_demand)
    mean_all  = sum(k * pmfs_all[j][k+1] for k in 0:inst.max_demand)
    if !isapprox(mean_none, mean_all; atol=1e-3)
        any_change = true
        break
    end
end
@assert any_change "DDU has no effect: all customer means identical under no-facilities vs all-facilities"
println("Test 5 passed: DDU effect is non-trivial (at least one customer mean changes)")

println("All distribution tests passed.")
