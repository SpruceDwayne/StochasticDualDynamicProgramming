using LinearAlgebra
using JuMP
using Printf
using Random

# ============================================================================
# DDU Indexing Convention (read before editing this file)
# ============================================================================
#
# NOTATION
#   ζ_t  ("zeta")   = incoming distribution context at stage t
#                     = outgoing region chosen at stage t-1
#                     = determines which distribution ξ_t is drawn from (μ_{ζ_t})
#
#   δ_t  ("delta")  = chosen decision region at stage t
#                     = the d ∈ D_t s.t. (x_t_next, y_t) ∈ X_t^d
#                     = determined by optimisation (read from 𝟙_d variables)
#                     = becomes ζ_{t+1}
#
# VALUE FUNCTION STORAGE
#   V[t][d]  where d ∈ D_t (outgoing regions at stage t)
#
#   Meaning : lower approximation to  E^d[Q_{t+1}(x_t, ξ_{t+1})]
#             = expected future value from stage t+1 onward,
#               given that the stage-t decision produced output state x_t
#               AND chose outgoing region d.
#
#   Read by : stage-t model  — epigraph constraints on θ_t:
#               θ_t ≥ α_i + β_i' x_next + M*(𝟙_d - 1)
#             activated by 𝟙_d (the outgoing-region indicator in stage-t model).
#
#   Written by : backward pass step t
#               — using trial point (x_leaving[s][t], δ_hist[s][t])
#               — by solving stage-(t+1) subproblems for scenarios from μ_{δ*}
#
# KEY DIFFERENCE FROM EXISTING CONVENTION
#   Existing : V[t] is a function of the *entering* state at stage t,
#              written by backward pass at stage t (solving stage-t models),
#              read by stage-(t-1) model.
#
#   DDU      : V[t][d] is a function of the *leaving* state at stage t,
#              written by backward pass step t (solving stage-(t+1) models),
#              read by stage-t model.
#
#   Consequence for the backward loop: at step t, we fix the stage-(t+1) model
#   state to x_leaving[s][t] (the output of the stage-t forward solve), NOT to
#   x_entering[s][t].  This is the most common source of indexing bugs.
#
# ============================================================================

# ============================================================================
# STAGE BUILDER CONTRACT FOR DDU
# ============================================================================
#
# Every user-supplied stg.build(t, vf_next, ω; fix_state) must satisfy:
#
# RETURN VALUES
#   (model, x_state, θ, misc)
#
#   x_state : Vector{JuMP.VariableRef}
#     The ENTERING-STATE variables.  These are the variables that
#     get_or_build_ddu_model! will fix to x_support (the parent state).
#
#     - For non-initial stages (t ≥ 2): x_state should contain the
#       SDDiP copy variables z_vars (or whatever encodes the parent
#       state in the model).  Typically left empty if z_vars is used.
#     - For stage 1 (no parent state): return JuMP.VariableRef[].
#       If x_state contains a decision variable (e.g. x1), that variable
#       will be FIXED to x0, preventing stage 1 from freely optimising.
#       This is the most common DDU stage-1 mistake.
#
#   θ : JuMP.VariableRef
#     The continuation epigraph variable.  Must appear in the objective
#     (typically as +θ in a minimisation).  Should have a finite lower
#     bound (e.g. θ >= -M_big) so the model is bounded when no cuts
#     are active.  Do NOT add cuts from vf_next inside the builder —
#     get_or_build_ddu_model! adds big-M cuts externally.
#
#   misc : Dict{Symbol, Any}
#     Required keys:
#       :region_indicators  => Dict{Int, JuMP.VariableRef}
#           Maps each outgoing region id d ∈ D_t to its binary indicator
#           variable 𝟙_d.  Must satisfy:
#             (i)  sum(𝟙_d for d in D_t) == 1  (exactly-one constraint)
#             (ii) 𝟙_d = 1  iff  (x_next, y) ∈ X_t^d  (membership)
#           For terminal stages (no outgoing regions): return Dict{Int,VariableRef}()
#
#       :x_next  => Vector{JuMP.VariableRef}
#           The OUTPUT state variables (the state that becomes the
#           entering state for stage t+1).  These appear in big-M cuts:
#             θ ≥ α_i + β_i' x_next + M*(𝟙_d - 1)
#
#     For non-initial stages using SDDiP:
#       :z_vars  => Vector{JuMP.VariableRef}
#           The continuous COPY variables z ∈ [0,1]^m that copy the
#           parent binary state.  These get fixed to x_support by
#           get_or_build_ddu_model! and are used by compute_sddip_cut!
#           for Lagrangian dual computation.
#           Include an initial equality constraint: z == fix_state
#           (this constraint will be found and replaced by JuMP.fix).
#
# STAGE SAMPLER AND CHILDREN
#   stg.sampler(ζ::Int)      → scenario ω ~ μ_{ζ}
#   stg.children(t, ζ::Int)  → (Xi, pXi) where Xi = support(μ_{ζ}),
#                               pXi = probabilities, sum(pXi) == 1
#   For stage 1: sampler(ζ_init) may return nothing (deterministic stage).
#   For terminal stage T: children is not called by the backward pass.
#
# stg.next_ctx  is a no-op placeholder in DDU (returning ζ is fine).
#
# WHAT NOT TO DO IN THE BUILDER
#   - Do not add cuts from vf_next (the vf_next argument is ignored by DDU).
#   - Do not include θ as a fixed variable (θ == 0) unless this is a
#     terminal stage where no continuation is expected.
#   - Do not put decision variables (x1, x2, ...) in x_state for DDU;
#     use z_vars for SDDiP copy variables instead.
#
# ============================================================================

# ============================================================================
# Types
# ============================================================================

"""
    DDURegion

Stochastic metadata for one cell of the decision-space partition at a stage.

Fields:
- `id`   : integer identifier (must be unique within a stage's region list)
- `Xi`   : finite support of the conditional distribution μ_d (vector of scenarios)
- `pXi`  : corresponding probabilities (must sum to 1)

NOTE: Region *membership* constraints (the JuMP constraints linking decision
variables to 𝟙_d) are NOT stored here.  They belong in the user-supplied
stage builder, which must also expose `misc[:region_indicators]`.
"""
struct DDURegion
    id::Int
    Xi::Vector{Any}
    pXi::Vector{Float64}
    function DDURegion(id, Xi, pXi)
        @assert length(Xi) == length(pXi) "Xi and pXi must have equal length"
        @assert abs(sum(pXi) - 1.0) < 1e-10 "pXi must sum to 1; got $(sum(pXi))"
        new(id, collect(Any, Xi), collect(Float64, pXi))
    end
end

"""
    DDUModelCache{T}

Per-(stage, scenario) model cache for DDU subproblems.

Extends the spirit of ModelCache{T} but tracks cut counts separately per
outgoing region, because each region's cuts carry a different 𝟙_d activation.

Field `last_cut_count_by_region[d_id]` = number of cuts from V[t][d_id] that
have already been added as big-M constraints to this cached model.
"""
mutable struct DDUModelCache{T}
    model::Union{Nothing, JuMP.Model}
    x_state::Union{Nothing, Vector{JuMP.VariableRef}}   # entering-state vars (fixed)
    x_next::Union{Nothing, Vector{JuMP.VariableRef}}    # leaving-state vars (in cuts)
    z_vars::Union{Nothing, Vector{JuMP.VariableRef}}    # SDDiP copy vars
    θ::Union{Nothing, JuMP.VariableRef}                 # continuation epigraph var
    region_indicators::Union{Nothing, Dict{Int, JuMP.VariableRef}}  # 𝟙_d → var
    misc::Union{Nothing, Dict{Symbol, Any}}
    last_cut_count_by_region::Dict{Int, Int}
    state_fixed::Bool
end

DDUModelCache{T}() where {T} = DDUModelCache{T}(
    nothing, nothing, nothing, nothing, nothing, nothing, nothing,
    Dict{Int,Int}(), false
)

"""
    DDUSDDP

Algorithm container for Decision-Dependent Uncertainty SDDiP.

Fields:
- `stages`   : stage definitions (same Stage struct as SDDP/MarkovSDDP)
- `V`        : V[t][d] = ValueFn for stage t, outgoing region d ∈ D_t
- `T`        : number of stages
- `γ`        : discount factor
- `M_big`    : big-M constant for cut activation.

               REQUIREMENT: M_big must satisfy
                 M_big > max_{x,d,i}(α^{d,i} + β^{d,i}' x) - lb(θ)
               where lb(θ) is the lower bound on θ in the subproblem model.

               In practice: M_big >> max_stage_cost / (1-γ).
               For a finite-horizon problem with T stages and max per-stage
               cost C: M_big = T * C * 10 is a conservative starting point.

               FAILURE MODE (silent!): if M_big is too small, the constraint
                 θ ≥ α + β' x_next + M_big*(𝟙_d - 1)
               for an INACTIVE region d (𝟙_d = 0) may still bind at some x,
               cutting off feasible solutions. The solver will appear to
               converge (cuts stop being generated) but the policy is
               suboptimal. There is no runtime error.

               To diagnose: compare the DDU policy value against an
               extensive-form solution (solve_extensive_control or manual LP).
               If the DDU value is strictly higher than the LP optimum,
               M_big is likely too small.
- `regions`  : regions[t] = Vector{DDURegion} listing all d ∈ D_t
- `model_cache` : model_cache[t][ω] = DDUModelCache, keyed by scenario ω
"""
mutable struct DDUSDDP
    stages::Vector{Stage}
    V::Vector{Dict{Int, ValueFn{Float64}}}
    T::Int
    γ::Float64
    M_big::Float64
    regions::Vector{Vector{DDURegion}}
    model_cache::Vector{Dict{Any, DDUModelCache{Float64}}}
end

function DDUSDDP(stages::Vector{Stage}, regions::Vector{Vector{DDURegion}};
                 discount::Float64 = 1.0, M_big::Float64 = 1e6)
    T = length(stages)
    @assert length(regions) == T "regions must have one entry per stage"
    V     = [Dict{Int, ValueFn{Float64}}() for _ in 1:T]
    cache = [Dict{Any, DDUModelCache{Float64}}() for _ in 1:T]
    DDUSDDP(stages, V, T, discount, M_big, regions, cache)
end

"""
    get_V_ddu!(m::DDUSDDP, t, d_id) -> ValueFn{Float64}

Lazily create and return V[t][d_id].  Same pattern as get_V! for MarkovSDDP.
"""
function get_V_ddu!(m::DDUSDDP, t::Int, d_id::Int)
    dict = m.V[t]
    if !haskey(dict, d_id)
        dict[d_id] = ValueFn{Float64}()
    end
    return dict[d_id]
end

# ============================================================================
# Forward record
# ============================================================================

"""
    DDUForwardRecord

Stores the trajectory of states and region choices from a DDU forward pass.

Fields (all indexed [sample_index][stage_index]):
- `x_entering` : x_entering[s][t] = state *entering* stage t  (= x_{t-1} in math)
                  This is what the stage-t model fixes as parent state.
- `x_leaving`  : x_leaving[s][t]  = state *leaving* stage t   (= x_t in math)
                  This is the trial point used in the backward pass at step t.
                  IMPORTANT: this is x_entering[s][t+1] for t < T.
- `δ_hist`     : δ_hist[s][t]  = outgoing region chosen at stage t (δ_t)
- `ζ_hist`     : ζ_hist[s][t]  = incoming context at stage t (ζ_t = δ_{t-1})

Indexing note: backward pass step t uses (x_leaving[s][t], δ_hist[s][t])
as the trial point, solving stage-(t+1) subproblems.
"""
struct DDUForwardRecord
    x_entering::Vector{Vector{Vector{Float64}}}
    x_leaving::Vector{Vector{Vector{Float64}}}
    δ_hist::Vector{Vector{Int}}
    ζ_hist::Vector{Vector{Int}}
end

# ============================================================================
# Model caching for DDU
# ============================================================================

"""
    _read_active_region(region_indicators) -> Int

Read the active region from a solved DDU model.  Uses argmax to be robust
against MIP solver tolerances (value may be 0.9999 rather than 1.0).

Returns 0 if region_indicators is empty (terminal stage with no outgoing regions).
"""
function _read_active_region(region_indicators::Dict{Int, JuMP.VariableRef})
    isempty(region_indicators) && return 0   # sentinel for terminal / no-region stage
    best_id  = first(keys(region_indicators))
    best_val = -Inf
    for (id, var) in region_indicators
        v = JuMP.value(var)
        if v > best_val
            best_val = v
            best_id  = id
        end
    end
    return best_id
end

"""
    get_or_build_ddu_model!(m::DDUSDDP, t, ω, x_support)

Retrieve or build the cached DDU subproblem for stage t, scenario ω,
with entering state fixed to x_support.

On first call:
  - Calls stg.build(t, ValueFn{Float64}(), ω; fix_state=x_support) to get
    the base model.  The builder must expose misc[:region_indicators].
  - Removes equality constraints that fix state (replaced by JuMP.fix).
  - Initialises last_cut_count_by_region to 0 for every region in m.regions[t].

On subsequent calls:
  - Re-fixes state variables to x_support.
  - For each outgoing region d in m.regions[t], adds any new cuts from
    V[t][d] as big-M constraints:
        θ ≥ α_i + β_i' x_next + M_big*(𝟙_d - 1)
  - Updates last_cut_count_by_region[d.id].

Returns (model, x_state, θ, misc).

NOTE: The cache key is ω alone (not (ζ, ω)), because the stage-t model
structure does not depend on the incoming context ζ_t — only which scenario ω
is used and which cuts have been added.
"""
function get_or_build_ddu_model!(m::DDUSDDP, t::Int, ω, x_support::AbstractVector{<:Real})
    stg = m.stages[t]

    # Cache key: ω (scenario determines model RHS; context ζ does not)
    cache_key = ω
    cache_dict = m.model_cache[t]
    if !haskey(cache_dict, cache_key)
        cache_dict[cache_key] = DDUModelCache{Float64}()
    end
    cache = cache_dict[cache_key]

    # ── First call: build ────────────────────────────────────────────────────
    if cache.model === nothing
        # Pass empty ValueFn — DDU manages cut addition externally below
        model, x_state, θ, misc = stg.build(t, ValueFn{Float64}(), ω; fix_state = x_support)

        @assert haskey(misc, :region_indicators) "DDU stage builder at t=$t must expose misc[:region_indicators]"

        region_indicators = misc[:region_indicators] :: Dict{Int, JuMP.VariableRef}

        # Convert any equality-based state fixing to JuMP.fix (same as existing code)
        vars_to_fix = haskey(misc, :z_vars) ? misc[:z_vars] : x_state
        find_and_remove_state_constraints!(model, vars_to_fix, x_support)
        for i in eachindex(vars_to_fix)
            JuMP.fix(vars_to_fix[i], x_support[i]; force = true)
        end

        x_next = haskey(misc, :x_next) ? misc[:x_next] : x_state

        cache.model             = model
        cache.x_state           = x_state
        cache.x_next            = x_next
        cache.z_vars            = haskey(misc, :z_vars) ? misc[:z_vars] : nothing
        cache.θ                 = θ
        cache.region_indicators = region_indicators
        cache.misc              = misc
        cache.state_fixed       = true

        # Initialise per-region cut counters to 0
        for d in m.regions[t]
            cache.last_cut_count_by_region[d.id] = 0
        end

        return model, x_state, θ, misc
    end

    # ── Subsequent calls: update ─────────────────────────────────────────────
    model             = cache.model
    x_state           = cache.x_state
    θ                 = cache.θ
    misc              = cache.misc
    region_indicators = cache.region_indicators
    x_next            = cache.x_next

    # Re-fix state (z_vars for SDDiP, x_state for standard)
    vars_to_fix = cache.z_vars !== nothing ? cache.z_vars : x_state
    for i in eachindex(vars_to_fix)
        JuMP.fix(vars_to_fix[i], x_support[i]; force = true)
    end

    # Add new big-M cuts from each region's ValueFn
    n_next = length(x_next)
    for d in m.regions[t]
        d_id = d.id
        # V[t][d] may not exist yet if no cuts have been generated for this region
        if !haskey(m.V[t], d_id)
            continue
        end
        vf = m.V[t][d_id]
        n_current = length(vf.cuts)
        last      = get(cache.last_cut_count_by_region, d_id, 0)

        if n_current > last
            ind_var = region_indicators[d_id]
            for i in (last + 1):n_current
                c = vf.cuts[i]
                # Big-M cut: active (tight) when 𝟙_d = 1, relaxed by M when 𝟙_d = 0
                @constraint(model,
                    θ >= c.α + sum(c.β[j] * x_next[j] for j in 1:n_next) +
                         m.M_big * (ind_var - 1))
            end
            cache.last_cut_count_by_region[d_id] = n_current
        end
    end

    return model, x_state, θ, misc
end

# ============================================================================
# Grouping helper
# ============================================================================

"""
    group_by_ddu_node(fwd::DDUForwardRecord, t) -> Dict

Group sample indices at backward step t by their trial point (δ_t, x_leaving_t).

Key = (δ_hist[s][t], Tuple(round.(Int, x_leaving[s][t])))

Rationale: two paths with the same outgoing region δ* but different leaving
states x* give cuts with different slopes β, so both groups must be processed.
Binary-state assumption (A12) ensures the x component is integer-valued and
the number of groups is finite.
"""
function group_by_ddu_node(fwd::DDUForwardRecord, t::Int)
    buckets = Dict{Tuple{Int, Tuple}, Vector{Int}}()
    S = length(fwd.x_leaving)
    for s in 1:S
        δ  = fwd.δ_hist[s][t]
        x  = fwd.x_leaving[s][t]
        xk = Tuple(round.(Int, x))
        key = (δ, xk)
        push!(get!(buckets, key, Int[]), s)
    end
    return buckets
end

# ============================================================================
# Forward pass
# ============================================================================

"""
    forward_pass_ddu_online!(m::DDUSDDP; S, x0, ζ_init) -> DDUForwardRecord

DDU online forward pass.

At each stage t:
  1. Sample ξ_t ~ μ_{ζ_t} via stg.sampler(ζ_t).
  2. Solve stage-t DDU model (which contains big-M cuts from V[t][d]).
  3. Read output state x_t from misc[:x_next].
  4. Read active outgoing region δ_t from misc[:region_indicators] via argmax.
  5. Set ζ_{t+1} = δ_t for the next stage.

Records:
  x_entering[s][t] = state entering stage t  (fixed as parent state in model)
  x_leaving[s][t]  = state leaving stage t   (output of stage-t decision)
  δ_hist[s][t]     = outgoing region chosen at stage t
  ζ_hist[s][t]     = incoming context at stage t

Note: x_leaving[s][t] == x_entering[s][t+1] for t < T.
"""
function forward_pass_ddu_online!(m::DDUSDDP;
                                   S::Int,
                                   x0::Vector{Float64},
                                   ζ_init::Int)
    T = m.T
    d = m.stages[1].state_dim

    x_entering = [[zeros(Float64, m.stages[t].state_dim) for t in 1:T] for _ in 1:S]
    x_leaving  = [[zeros(Float64, m.stages[t].state_dim) for t in 1:T] for _ in 1:S]
    δ_hist     = [[0 for _ in 1:T] for _ in 1:S]
    ζ_hist     = [[0 for _ in 1:T] for _ in 1:S]

    for s in 1:S
        x = copy(x0)
        ζ = ζ_init   # incoming context for stage 1

        for t in 1:T
            stg = m.stages[t]

            # Record entering state and context
            x_entering[s][t] = copy(x)
            ζ_hist[s][t]     = ζ

            # Sample from the distribution selected by the incoming context
            ωt = stg.sampler(ζ)

            # Solve stage-t DDU subproblem
            model, x_state, θ, misc = get_or_build_ddu_model!(m, t, ωt, x)
            JuMP.optimize!(model)

            status = JuMP.termination_status(model)
            status == MOI.OPTIMAL || @warn "DDU forward pass: stage $t solver status = $status"

            # Read leaving state
            x_next_vars = misc[:x_next]
            x_out = JuMP.value.(x_next_vars)
            x_leaving[s][t] = copy(x_out)

            # Read active outgoing region (argmax is robust to MIP tolerances)
            ri = cache_region_indicators(misc)
            δt = _read_active_region(ri)

            δ_hist[s][t] = δt

            # Advance
            x = x_out
            ζ = δt   # outgoing becomes next incoming
        end
    end

    return DDUForwardRecord(x_entering, x_leaving, δ_hist, ζ_hist)
end

# Helper: fetch region_indicators from misc, with a clear error
function cache_region_indicators(misc::Dict{Symbol,Any})
    @assert haskey(misc, :region_indicators) "stage builder must store misc[:region_indicators]::Dict{Int,VariableRef}"
    return misc[:region_indicators] :: Dict{Int, JuMP.VariableRef}
end

# ============================================================================
# Backward pass
# ============================================================================

"""
    backward_pass_ddu_sddip!(m::DDUSDDP; fwd, config, iter, force_every, atol)

DDU SDDiP backward pass.

For each backward step t in (T-1):-1:1:
  - Groups trial points by (δ_hist[s][t], x_leaving[s][t]).
  - For each group (δ*, x*):
      * Fetches scenarios from μ_{δ*}: (Xi, ps) = children(t, δ*) via the
        DDURegion for δ* in m.regions[t].
      * For each scenario (ωj, pj): solves the stage-(t+1) DDU subproblem
        with entering state fixed to x*  (NOTE: x* = x_leaving[s][t], not
        x_entering[s][t] — this is the critical DDU indexing difference).
      * Computes SDDiP Lagrangian cuts via compute_sddip_cut!.
      * Aggregates expected cut: ᾱ = Σ pj αj, β̄ = Σ pj βj.
  - Adds cut (ᾱ, β̄) to V[t][δ*] if it improves the current envelope.
  - V[t][δ*] cuts will be picked up as big-M constraints in stage-t models
    on the next call to get_or_build_ddu_model!(m, t, ·).

Requirement: every stage builder (for t ≥ 2) must expose misc[:z_vars].
"""
function backward_pass_ddu_sddip!(
    m::DDUSDDP;
    fwd::DDUForwardRecord,
    config::SDDiPConfig,
    iter::Int        = 1,
    force_every::Int = 10,
    atol::Float64    = 1e-8,
)
    T           = m.T
    n_cut_types = config.cut_type === :SB_IO ? 2 : 1

    # Build a lookup: region_id -> DDURegion, per stage
    region_map = [Dict(r.id => r for r in m.regions[t]) for t in 1:T]

    # Backward loop: step t computes cuts for V[t][d] by solving stage-(t+1)
    for t in (T-1):-1:1
        stg_next = m.stages[t + 1]

        # Group trial points: key = (δ_hist[s][t], x_leaving[s][t] rounded to Int)
        buckets = group_by_ddu_node(fwd, t)

        for ((δ_star, _xkey), scen_idx) in buckets
            s₁      = first(scen_idx)
            # x* = the state *leaving* stage t = entering state for stage t+1
            # This is the DDU trial point — NOT x_entering[s][t].
            x_star  = fwd.x_leaving[s₁][t]

            # Fetch the conditional distribution for outgoing region δ*
            if !haskey(region_map[t], δ_star)
                @warn "backward_pass_ddu_sddip!: δ_star=$δ_star not in m.regions[$t]; skipping"
                continue
            end
            region = region_map[t][δ_star]
            Xi, ps = region.Xi, region.pXi

            α_accs = zeros(Float64, n_cut_types)
            β_accs = [zeros(Float64, stg_next.state_dim) for _ in 1:n_cut_types]

            for k in eachindex(Xi)
                ωj = Xi[k]
                pj = ps[k]

                # Solve stage-(t+1) subproblem: state fixed to x_star, scenario ωj
                model, _, _, misc_next = get_or_build_ddu_model!(m, t + 1, ωj, x_star)
                haskey(misc_next, :z_vars) || continue
                z_vars = misc_next[:z_vars]

                pairs = compute_sddip_cut!(model, z_vars, collect(Float64, x_star), config)
                for (ci, (α, β)) in enumerate(pairs)
                    α_accs[ci]   += pj * α
                    β_accs[ci]  .+= pj .* β
                end
            end

            # Add accumulated cuts to V[t][δ*]
            vf = get_V_ddu!(m, t, δ_star)
            for ci in 1:n_cut_types
                val_old, _  = evaluate(vf, x_star)
                val_new      = α_accs[ci] + dot(β_accs[ci], x_star)
                should_force = force_every > 0 && (iter % force_every == 0)
                if should_force || (val_new > val_old + atol)
                    add_cut!(vf, α_accs[ci], β_accs[ci], t)
                end
            end
        end
    end
    return nothing
end

# ============================================================================
# Run loop
# ============================================================================

"""
    compute_ddu_lb!(m::DDUSDDP, x0, ζ_init) -> Float64

Compute the DDU-SDDiP lower bound: the optimal value of the stage-1 MILP
with all currently accumulated big-M cuts.

This is the direct analogue of the stage-1 LP lower bound in standard SDDP.
Because the stage-1 MILP optimises jointly over all outgoing regions (via
the binary δ indicators and big-M cuts from every V[1][d]), it correctly
accounts for all regions that could be active at the optimum — not just one
pre-selected region.

The model is retrieved from (or built into) the DDU cache, so the call is
cheap after the first build.
"""
function compute_ddu_lb!(m::DDUSDDP, x0::AbstractVector{<:Real}, ζ_init::Int)
    stg = m.stages[1]
    ω1  = stg.sampler(ζ_init)
    model, _, _, _ = get_or_build_ddu_model!(m, 1, ω1, collect(Float64, x0))
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    status == MOI.OPTIMAL || @warn "compute_ddu_lb!: stage-1 solver status = $status"
    return JuMP.objective_value(model)
end

"""
    run_ddu_sddip!(m::DDUSDDP; x0, ζ_init, config, S, max_iter, patience,
                   lb_tol, rng, force_every, cut_atol, logfn) -> NamedTuple

DDU SDDiP training loop.

Keyword arguments mirror run_sddip! / run_markov_sddip!.

- `ζ_init`    : incoming distribution context at stage 1 (Int, required)
- `lb_tol`    : stop early if the lower bound improves by less than this
                between consecutive iterations (analogous to `value_tol` in
                standard SDDP); 0.0 disables this check.

Convergence monitoring uses the **stage-1 MILP lower bound** (via
`compute_ddu_lb!`) — the direct analogue of the stage-1 LP bound in standard
SDDP.  This integrates all regions through the joint big-M optimisation and
does not require pre-selecting a single region or state to track.

Cut counting aggregates across all stages and all regions.
Stopping rules: lower-bound stagnation (patience) and optional lb_tol.
"""
function run_ddu_sddip!(
    m::DDUSDDP;
    x0::AbstractVector,
    ζ_init::Int,
    config::SDDiPConfig   = SDDiPConfig(),
    S::Int                = 1,
    max_iter::Int         = 1_000,
    patience::Int         = 20,
    lb_tol::Float64       = 0.0,
    rng                   = Random.default_rng(),
    force_every::Int      = 10,
    cut_atol::Float64     = 1e-8,
    logfn                 = nothing,
)
    total_cuts() = sum(
        t -> sum(vf -> length(vf.cuts), values(m.V[t]); init = 0),
        1:m.T; init = 0,
    )
    cuts_by_stage() = [
        sum(vf -> length(vf.cuts), values(m.V[t]); init = 0)
        for t in 1:m.T
    ]

    prev_total = total_cuts()
    stagnant   = 0
    x0f        = collect(Float64, x0)
    prev_lb    = compute_ddu_lb!(m, x0f, ζ_init)
    hist       = Vector{NamedTuple}()

    Random.seed!(rng, rand(UInt))

    for it in 1:max_iter
        cfg_it = _active_config(config, it)
        fwd = forward_pass_ddu_online!(m; S = S, x0 = x0f, ζ_init = ζ_init)
        backward_pass_ddu_sddip!(m; fwd = fwd, config = cfg_it, iter = it,
                                  force_every = force_every, atol = cut_atol)

        cur_total = total_cuts()
        new_cuts  = cur_total - prev_total
        prev_total = cur_total

        # Lower bound: stage-1 MILP objective with all current cuts.
        # Analogous to the stage-1 LP bound in standard SDDP.
        cur_lb = compute_ddu_lb!(m, x0f, ζ_init)
        Δlb    = cur_lb - prev_lb
        prev_lb = cur_lb

        stats = (iter = it, new_cuts = new_cuts, total_cuts = cur_total,
                 lb = cur_lb, Δlb = Δlb, per_stage = cuts_by_stage(),
                 phase = it <= config.burnin_iters ? :burnin : :main)
        push!(hist, stats)

        phase_tag = it <= config.burnin_iters ? "[SB burn-in] " : ""
        if logfn === nothing
            @printf "iter %4d | %snew cuts: %2d | total: %3d | LB=%.6f | ΔLB=%.3e\n" it phase_tag new_cuts cur_total cur_lb Δlb
        else
            logfn(it, stats)
        end

        stagnant = (new_cuts == 0) ? (stagnant + 1) : 0
        if stagnant >= patience
            println("DDU early stop: no new cuts for $patience consecutive iterations.")
            return (iters = it, cuts_per_stage = cuts_by_stage(), history = hist,
                    lb = cur_lb)
        end
        if lb_tol > 0 && abs(Δlb) ≤ lb_tol
            println("DDU early stop: |ΔLB| ≤ $lb_tol.")
            return (iters = it, cuts_per_stage = cuts_by_stage(), history = hist,
                    lb = cur_lb)
        end
    end

    println("DDU reached max_iter without early stop.")
    return (iters = max_iter, cuts_per_stage = cuts_by_stage(), history = hist,
            lb = prev_lb)
end
