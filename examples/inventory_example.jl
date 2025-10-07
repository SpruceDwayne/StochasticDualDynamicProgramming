using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
Pkg.instantiate()

using JuMP, HiGHS
using SDDPBAPE
using Random
vf = ValueFn{Float64}()   # works
"Simple discrete sampler with equal weights."
struct DiscreteSampler{T}
    support::Vector{T}
end
sample(ds::DiscreteSampler) = ds.support[rand(1:length(ds.support))]
weight(ds::DiscreteSampler, ω) = 1.0/length(ds.support)
all_support(ds::DiscreteSampler) = ds.support

"Builder for the inventory stage."
function build_inventory_stage(t, vf_next::ValueFn, ω;
                               fix_state::Union{Nothing,Vector{Float64}}=nothing,
                               c=1.0, h=0.1, A=1.0)  # A maps x_t → x_{t+1} via +1
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # State and decisions
    @variable(model, x_t)          # state before order
    @variable(model, q_t >= 0)
    @variable(model, x_next)
    @variable(model, xplus >= 0)   # linearize holding: x_next = xplus - xminus (skip xminus if backlogging disallowed)
    @variable(model, θ)            # cost-to-go

    # Fix x_t if in backward pass
    if fix_state !== nothing
        @constraint(model, x_t == fix_state[1])
    end

    # Inventory balance with demand ω (treat unmet demand as backorder allowed via x_next free)
    @constraint(model, x_next == x_t + q_t - ω)
    @constraint(model, xplus >= x_next)  # xplus ≥ x_next and xplus ≥ 0 → xplus = max(x_next, 0)

    # Add next-stage cuts: θ ≥ α + β * x_next  (if no cuts, θ ≥ 0 by convention)
    if !isempty(vf_next.cuts)
        for c in vf_next.cuts
            @constraint(model, θ >= c.α + c.β[1] * x_next)
        end
    else
        @constraint(model, θ >= 0)  # terminal if no cuts yet
    end

    # Objective: purchase + holding + next cost
    @objective(model, Min, c*q_t + h*xplus + θ)

    # Provide β w.r.t. x_t via chain rule x_next = 1 * x_t + ...
    # Active β_{t+1} is the gradient of the cut that binds θ. We don’t know it pre-solve,
    # so after solving we compute β_xt = 1 * β_next.
    misc = Dict{Symbol,Any}()

    function finalize!(model)
        # pick an active β_next: if multiple are tight, any convex comb is valid; we just pick first
        βnext = 0.0
        if !isempty(vf_next.cuts)
            best = -Inf
            for c in vf_next.cuts
                val = c.α + c.β[1] * value(x_next)
                if val > best + 1e-8
                    best = val
                    βnext = c.β[1]
                end
            end
        else
            βnext = 0.0
        end
        misc[:β_xt] = [βnext]  # because x_next = 1 * x_t + ...
    end

    return model, [x_t], θ, merge(misc, Dict(:finalize! => finalize!, :x_next => x_next))
end

"Helper that runs finalize! after optimize! if present."
function optimize!(tpl)
    model, x_t, θ, misc = tpl
    JuMP.optimize!(model)
    if haskey(misc, :finalize!)
        misc[:finalize!](model)
    end
    return
end


# Problem data
T      = 5
demands = [4.0, 6.0, 8.0]            # support
sampler = DiscreteSampler(demands)

# Stage factory
stages = Vector{Stage}(undef, T)
for t in 1:T
    build = (t, vf_next, ω; fix_state=nothing) ->
        build_inventory_stage(t, vf_next, ω; fix_state=fix_state)
    stages[t] = Stage(t, 1,
                      build,
                      () -> sample(sampler),
                      ω -> weight(sampler, ω))
end

algo = SDDP(stages; discount=1.0)

# Seed a trivial terminal cut V_{T+1}(x) = 0 by leaving algo.V[T] to be learned.
# Initialize with one forward pass to create trial points
for it in 1:10
    forward_pass!(algo)
    backward_pass!(algo; batch_size=1)  # use more samples for better stability
end

println("Stage 1 has $(length(algo.V[1].cuts)) cuts.")
