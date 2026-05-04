include(joinpath(@__DIR__, "..", "problem_src", "data.jl"))

const INSTANCE_PATH = joinpath(@__DIR__, "..", "..", "instance_data.json")

inst = load_instance(INSTANCE_PATH)

# Test 1: facility_zones covers all num_zones zones and all 8 facilities
@assert length(inst.facility_zones) == inst.num_facilities "facility_zones length mismatch"
@assert sort(unique(inst.facility_zones)) == collect(1:inst.num_zones) "facility_zones zone coverage mismatch"
println("Test 1 passed: facility_zones has $(inst.num_facilities) entries covering zones 1:$(inst.num_zones)")

# Test 2: profit_matrix is (num_facilities × num_customers), values in plausible range
@assert size(inst.profit_matrix) == (inst.num_facilities, inst.num_customers) "profit_matrix size mismatch: $(size(inst.profit_matrix))"
pmin, pmax = minimum(inst.profit_matrix), maximum(inst.profit_matrix)
@assert pmin > 0 && pmax < 500 "profit_matrix values out of expected range: [$pmin, $pmax]"
println("Test 2 passed: profit_matrix size=$(size(inst.profit_matrix)), min≈$(round(pmin;digits=1)), max≈$(round(pmax;digits=1))")

# Test 3: Customer 1 base mean recoverable from alpha/beta
base_mean_1 = inst.max_demand * inst.alpha[1] / (inst.alpha[1] + inst.beta[1])
@assert isapprox(base_mean_1, 1.5; atol=1e-6) "Customer 1 base mean mismatch: $base_mean_1"
println("Test 3 passed: customer 1 base mean = $base_mean_1")

# Test 4: activation_regions is a powerset of size 2^num_zones, region 1 = empty set
@assert length(inst.activation_regions) == 2^inst.num_zones "activation_regions size mismatch"
@assert inst.activation_regions[1] == Set{Int}() "region 1 should be empty set"
println("Test 4 passed: $(length(inst.activation_regions)) activation regions, region 1 = empty set")

# Test 5: zone_facilities covers every facility exactly once
all_facs = sort(vcat(collect(values(inst.zone_facilities))...))
@assert all_facs == collect(1:inst.num_facilities) "zone_facilities coverage mismatch"
println("Test 5 passed: zone_facilities covers all $(inst.num_facilities) facilities")

# Test 6: interaction_type is one of the supported types
@assert inst.interaction_type in ("A", "B", "C", "D") "Unknown interaction_type: $(inst.interaction_type)"
println("Test 6 passed: interaction_type = $(inst.interaction_type)")

println("All tests passed.")
