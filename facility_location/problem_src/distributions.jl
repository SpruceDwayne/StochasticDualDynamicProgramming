using Distributions
using Random

include(joinpath(@__DIR__, "data.jl"))

"""
    region_to_zone_active(inst, d) → Vector{Bool}

Return a length-`num_zones` Bool vector; entry z is true iff zone z is active in region d.
"""
function region_to_zone_active(inst::InstanceData, d::Int)::Vector{Bool}
    active = inst.activation_regions[d]
    return [z in active for z in 1:inst.num_zones]
end

# Internal helper: build the BetaBinomial distribution for customer j given zone_active.
function _customer_dist_old(inst::InstanceData, j::Int, zone_active::Vector{Bool})
    n = inst.max_demand   # 10
    s = 6#12#6                 # scale
    base_mean = n * inst.alpha[j] / (inst.alpha[j] + inst.beta[j])
    mean_j = base_mean
    for k in 1:inst.num_zones
        z = inst.zone_order[j][k]
        if zone_active[z]
            mean_j += (0.5^k) * base_mean
        end
    end
    mean_j = min(mean_j, n - 1e-6)
    p = mean_j / n
    return BetaBinomial(n, p * s, (1 - p) * s)
end

function _customer_dist(inst::InstanceData, j::Int, zone_active::Vector{Bool})
    n          = inst.max_demand
    base_mean  = n * inst.alpha[j] / (inst.alpha[j] + inst.beta[j])
    zone_order = inst.zone_order[j]
    itype      = inst.interaction_type

    mean_delta  = 0.0
    scale_delta = 0.0

    if itype == "A"
        for (n_idx, z) in enumerate(zone_order)
            α = inst.alpha_base ^ n_idx
            β = inst.beta_base  ^ n_idx
            active = zone_active[z] ? 1.0 : 0.0
            mean_delta  += α * active
            scale_delta += β * active
        end

    elseif itype == "B"
        active = zone_active[zone_order[1]] ? 1.0 : 0.0
        mean_delta  = inst.alpha_base * active
        scale_delta = inst.beta_base  * active

    elseif itype == "C"
        for (n_idx, z) in enumerate(zone_order)
            if zone_active[z]
                mean_delta  = inst.alpha_base ^ n_idx
                scale_delta = inst.beta_base  ^ n_idx
                break
            end
        end

    elseif itype == "D"
        for (n_idx, z) in enumerate(zone_order)
            α    = inst.alpha_base ^ n_idx
            β    = inst.beta_base  ^ n_idx
            sign = n_idx == 1 ? 1.0 : -1.0
            active = zone_active[z] ? 1.0 : 0.0
            mean_delta  += α * sign * active
            scale_delta += β * sign * active
        end
    end

    mean_j  = clamp(base_mean * (1.0 + mean_delta),  0.5, n - 1e-6)
    scale_j = max(inst.base_scale * (1.0 + scale_delta), 0.5)
    p = mean_j / n
    return BetaBinomial(n, p * scale_j, (1 - p) * scale_j)
end

"""
    compute_demand_distributions(inst, zone_active) → Vector{Vector{Float64}}

Return per-customer PMFs: `pmfs[j][k+1] = P(ξ_j = k)` for k ∈ 0:max_demand.
"""
function compute_demand_distributions(inst::InstanceData, zone_active::Vector{Bool})
    return [pdf.(_customer_dist(inst, j, zone_active), 0:inst.max_demand) for j in 1:inst.num_customers]
end

"""
    compute_demand_pmfs(inst, d) → Matrix{Float64}  size (num_customers, max_demand+1)

Row j is the PMF for customer j under region d.
"""
function compute_demand_pmfs(inst::InstanceData, d::Int)::Matrix{Float64}
    zone_active = region_to_zone_active(inst, d)
    pmfs = compute_demand_distributions(inst, zone_active)
    # each pmfs[j] is length 11; hcat gives (11 × 15), transpose → (15 × 11)
    return Matrix{Float64}(reduce(hcat, pmfs)')
end

"""
    identify_region(inst, x, b=zeros) → Int

Return the region index d matching the zone activation pattern induced by facility
configuration x. The optional b argument is accepted for interface compatibility
but zone activations are always computed from x.
"""
function identify_region(inst::InstanceData, x::Vector{Int},
                         b::Vector{Int}=zeros(Int, inst.num_zones))::Int
    active_set = Set{Int}(
        z for z in 1:inst.num_zones
        if any(x[i] == 1 for i in inst.zone_facilities[z])
    )
    for d in 1:length(inst.activation_regions)
        inst.activation_regions[d] == active_set && return d
    end
    error("No matching region found for zone activation $active_set")
end

"""
    sample_scenarios(inst, d; N=50, seed=42) → (Matrix{Int}, Vector{Float64})

Sample N independent demand vectors for region d.
Returns `(scenarios, probs)` where `scenarios` is `(N × num_customers)` and
`probs` is the uniform weight vector of length N.
"""
function sample_scenarios(inst::InstanceData, d::Int; N::Int=50, seed::Int=42)
    rng = MersenneTwister(seed)
    zone_active = region_to_zone_active(inst, d)
    scenarios = Matrix{Int}(undef, N, inst.num_customers)
    for j in 1:inst.num_customers
        scenarios[:, j] = rand(rng, _customer_dist(inst, j, zone_active), N)
    end
    return scenarios, fill(1.0 / N, N)
end

"""
    generate_saa_scenarios(inst; N=50, seed=42) → Dict{Int, Matrix{Int}}

Generate the fixed SAA scenario sets for all regions. Call once before the algorithm.
Each entry maps region d → `(N × num_customers)` integer matrix.
The RNG is seeded once and advanced in region order for reproducibility.
"""
function generate_saa_scenarios(inst::InstanceData; N::Int=50, seed::Int=42)::Dict{Int,Matrix{Int}}
    rng = MersenneTwister(seed)
    out = Dict{Int,Matrix{Int}}()
    for d in 1:length(inst.activation_regions)
        zone_active = region_to_zone_active(inst, d)
        mat = Matrix{Int}(undef, N, inst.num_customers)
        for j in 1:inst.num_customers
            mat[:, j] = rand(rng, _customer_dist(inst, j, zone_active), N)
        end
        out[d] = mat
    end
    return out
end
