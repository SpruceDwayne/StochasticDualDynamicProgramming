import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "SDDPBAPE"))
for pkg in ["Statistics", "Printf", "Distributions"]
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end

using Statistics, Printf, Distributions, Random

include(joinpath(@__DIR__, "distributions.jl"))

const INST_PATH = joinpath(@__DIR__, "..", "..", "instance_data_D.json")
const inst      = load_instance(INST_PATH)
const N_SAA     = 10
const saa       = generate_saa_scenarios(inst; N=N_SAA, seed=123)

nR = length(inst.activation_regions)
nJ = inst.num_customers

# ── True expected demand per (region, customer) ────────────────────────────────
true_mean = Matrix{Float64}(undef, nR, nJ)
for d in 1:nR
    za = region_to_zone_active(inst, d)
    for j in 1:nJ
        dist = _customer_dist(inst, j, za)
        true_mean[d, j] = mean(dist)
    end
end

# ── SAA mean demand per (region, customer) ─────────────────────────────────────
saa_mean = Matrix{Float64}(undef, nR, nJ)
for d in 1:nR
    saa_mean[d, :] = vec(mean(saa[d]; dims=1))
end

# ── Print header ───────────────────────────────────────────────────────────────
sep = "─"^(10 + 7*nJ)
println("\nInstance D | N_SAA=$N_SAA | seed=123")
println("Regions: $nR | Customers: $nJ | max_demand=$(inst.max_demand)")

println("\n" * "="^length(sep))
println("TABLE 1 — SAA mean demand  (rows=regions, cols=customers 1–$nJ)")
println("="^length(sep))
@printf("%-10s", "Region")
for j in 1:nJ; @printf("%7d", j); end
println()
println(sep)
for d in 1:nR
    zones = collect(inst.activation_regions[d])
    label = isempty(zones) ? "∅" : join(sort(zones), ",")
    @printf("d=%-2d {%-5s}", d, label)
    for j in 1:nJ; @printf("%7.2f", saa_mean[d, j]); end
    println()
end

println("\n" * "="^length(sep))
println("TABLE 2 — True expected demand  (rows=regions, cols=customers 1–$nJ)")
println("="^length(sep))
@printf("%-10s", "Region")
for j in 1:nJ; @printf("%7d", j); end
println()
println(sep)
for d in 1:nR
    zones = collect(inst.activation_regions[d])
    label = isempty(zones) ? "∅" : join(sort(zones), ",")
    @printf("d=%-2d {%-5s}", d, label)
    for j in 1:nJ; @printf("%7.2f", true_mean[d, j]); end
    println()
end

println("\n" * "="^length(sep))
println("TABLE 3 — SAA bias = saa_mean − true_mean  (positive = SAA overestimates)")
println("="^length(sep))
@printf("%-10s", "Region")
for j in 1:nJ; @printf("%7d", j); end
println("   |  max_abs  mean_abs")
println(sep * "─────────────────────")
for d in 1:nR
    zones = collect(inst.activation_regions[d])
    label = isempty(zones) ? "∅" : join(sort(zones), ",")
    bias  = saa_mean[d, :] .- true_mean[d, :]
    @printf("d=%-2d {%-5s}", d, label)
    for j in 1:nJ; @printf("%7.2f", bias[j]); end
    @printf("   | %8.3f  %8.3f\n", maximum(abs, bias), mean(abs, bias))
end

# ── Per-region aggregate ───────────────────────────────────────────────────────
println("\n" * "="^50)
println("TABLE 4 — Per-region aggregates")
println("="^50)
@printf("%-10s %12s %12s %12s %12s\n",
    "Region", "SAA_sum", "True_sum", "Bias_sum", "Max_abs_bias")
println("─"^50)
for d in 1:nR
    zones = collect(inst.activation_regions[d])
    label = isempty(zones) ? "∅" : join(sort(zones), ",")
    s_saa  = sum(saa_mean[d, :])
    s_true = sum(true_mean[d, :])
    bias   = saa_mean[d, :] .- true_mean[d, :]
    @printf("d=%-2d {%-5s} %12.2f %12.2f %12.3f %12.3f\n",
        d, label, s_saa, s_true, s_saa - s_true, maximum(abs, bias))
end
