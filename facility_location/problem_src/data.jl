using JSON

struct InstanceData
    num_customers::Int
    num_facilities::Int
    num_zones::Int
    max_demand::Int
    R::Int
    C::Int
    O::Int
    k::Int
    T::Int
    customer_coords::Matrix{Float64}
    facility_coords::Matrix{Float64}
    facility_zones::Vector{Int}
    profit_matrix::Matrix{Float64}
    dist_matrix::Matrix{Float64}
    alpha::Vector{Float64}
    beta::Vector{Float64}
    zone_order::Dict{Int,Vector{Int}}
    activation_regions::Vector{Set{Int}}
    zone_facilities::Dict{Int,Vector{Int}}
    interaction_type::String
    alpha_base::Float64
    beta_base::Float64
    base_scale::Float64
end

function load_instance(path::String)
    d = JSON.parsefile(path)

    facility_zones = [z + 1 for z in d["facility_zones"]]

    zone_order = Dict{Int,Vector{Int}}()
    for (k, v) in d["zone_order_per_customer"]
        zone_order[parse(Int, k) + 1] = [z + 1 for z in v]
    end

    activation_regions = [Set{Int}(z + 1 for z in region) for region in d["activation_regions"]]

    num_zones_val = d["num_zones"]
    fz_temp = [z + 1 for z in d["facility_zones"]]
    zone_facilities = Dict{Int,Vector{Int}}(z => Int[] for z in 1:num_zones_val)
    for (i, z) in enumerate(fz_temp)
        push!(zone_facilities[z], i)
    end

    # Each row in JSON corresponds to one facility; inner length = num_customers
    # reduce(hcat, rows) yields (num_customers × num_facilities), permutedims gives (num_facilities × num_customers)
    profit_matrix = permutedims(reduce(hcat, [Float64.(row) for row in d["profit_matrix"]]))
    dist_matrix   = permutedims(reduce(hcat, [Float64.(row) for row in d["dist_matrix"]]))

    customer_coords = permutedims(reduce(hcat, [Float64.(c) for c in d["customer_coords"]]))
    facility_coords = permutedims(reduce(hcat, [Float64.(f) for f in d["facility_coords"]]))

    return InstanceData(
        d["num_customers"],
        d["num_facilities"],
        d["num_zones"],
        d["max_demand"],
        d["R"],
        d["C"],
        d["O"],
        d["k"],
        d["T"],
        customer_coords,
        facility_coords,
        facility_zones,
        profit_matrix,
        dist_matrix,
        Float64.(d["customer_base_alpha"]),
        Float64.(d["customer_base_beta"]),
        zone_order,
        activation_regions,
        zone_facilities,
        d["interaction_type"],
        Float64(d["alpha_base"]),
        Float64(d["beta_base"]),
        Float64(d["base_scale"]),
    )
end
