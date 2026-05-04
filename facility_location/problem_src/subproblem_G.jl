using JuMP, Gurobi

"""
    make_stage_builder(inst::InstanceData, t::Int) -> Function

Factory that returns a stage-t DDU subproblem builder.

Signature of the returned function:
    (t, vf_next, ω; fix_state) -> (model, x_state, θ, misc)

where ω is a Vector{<:Real} of length num_customers (the demand realization).

Contract (DDU-SDDiP interface):
- Stage 1 (root):  x_state = VariableRef[]  (no parent state to fix)
- Stages 2–T:      x_state = z_vars         (continuous copies of entering state)
- Non-terminal:    misc[:region_indicators] has 8 binary δ[d] variables
- Terminal:        misc[:region_indicators] is empty; θ is fixed to 0
"""
function make_stage_builder(inst::InstanceData, t::Int, grb_env::Gurobi.Env)
    nI  = inst.num_facilities               # 8
    nJ  = inst.num_customers                # 15
    nZ  = inst.num_zones                    # 3
    nD  = length(inst.activation_regions)   # 8
    P   = inst.profit_matrix                # (nI × nJ) profit matrix
    O   = Float64(inst.O)                   # opening cost per facility
    C   = Float64(inst.C)                   # capacity per facility
    k   = inst.k                            # max new openings per stage (2)
    is_root     = (t == 1)
    is_terminal = (t == inst.T)
    # Lower bound on θ: max per-stage gross profit ≈ 394 × 120 ≈ 50 000
    theta_lb = is_terminal ? 0.0 : -Float64(inst.T - t) * 50_000.0

    function builder(_t, _vf, ω; fix_state)
        model = Model(() -> Gurobi.Optimizer(grb_env))
        set_silent(model)

        # ── Facility configuration (decision) ─────────────────────────────────
        @variable(model, x[1:nI], Bin)

        # ── Entering-state copy variables (non-root only) ─────────────────────
        # z ∈ [0,1]^nI copies x_{t-1}.  The initial equality z == fix_state is
        # found and removed by get_or_build_ddu_model!, then replaced by JuMP.fix.
        z_refs = JuMP.VariableRef[]
        if !is_root
            @variable(model, z[1:nI], lower_bound = 0.0, upper_bound = 1.0)
            for i in 1:nI
                @constraint(model, z[i] == fix_state[i])
            end
            append!(z_refs, z)

            # Monotonicity + per-stage opening budget
            for i in 1:nI
                @constraint(model, x[i] >= z_refs[i])
            end
            @constraint(model, sum(x[i] - z_refs[i] for i in 1:nI) <= k)

            # ── Demand served (non-root stages only) ──────────────────────────
            # Stage 1 is a pure investment stage: facilities are chosen but no
            # demand is realised yet.  Revenue and allocation start at stage 2.
            demand = Float64[ω[j] for j in 1:nJ]
            @variable(model, w[1:nI, 1:nJ] >= 0)
            for j in 1:nJ
                @constraint(model, sum(w[i, j] for i in 1:nI) <= demand[j])
            end
            for i in 1:nI
                @constraint(model, sum(w[i, j] for j in 1:nJ) <= C * z_refs[i])
            end
        else
            # Root stage: only the opening budget (x_prev = 0, monotonicity trivial)
            @constraint(model, sum(x) <= k)
        end

        # ── Zone activation + region indicators (non-terminal only) ───────────
        region_ind = Dict{Int, JuMP.VariableRef}()
        if !is_terminal
            @variable(model, a[1:nZ], Bin)
            @variable(model, δ[1:nD], Bin)

            for zz in 1:nZ
                fz = inst.zone_facilities[zz]
                @constraint(model, a[zz] <= sum(x[i] for i in fz))
                for i in fz
                    @constraint(model, a[zz] >= x[i])
                end
            end

            @constraint(model, sum(δ) == 1)
            for zz in 1:nZ
                @constraint(model,
                    a[zz] == sum(δ[d] for d in 1:nD if zz in inst.activation_regions[d]))
            end

            for d in 1:nD
                region_ind[d] = δ[d]
            end
        end

        # ── Continuation variable ─────────────────────────────────────────────
        if is_terminal
            @variable(model, θ == 0.0)
        else
            @variable(model, θ >= theta_lb)
        end

        # ── Objective ─────────────────────────────────────────────────────────
        # Root (stage 1): opening cost + future value only (no demand revenue).
        # Non-root: negative profit from serving demand + opening cost + future.
        if is_root
            @objective(model, Min, O * sum(x) + θ)
        else
            opening = O * sum(x[i] - z_refs[i] for i in 1:nI)
            @objective(model, Min,
                -sum(P[i, j] * w[i, j] for i in 1:nI, j in 1:nJ) + opening + θ)
        end

        # ── misc dict required by the DDU interface ────────────────────────────
        misc = Dict{Symbol, Any}(
            :x_next            => JuMP.VariableRef[x[i] for i in 1:nI],
            :region_indicators => region_ind,
        )
        if !is_root
            misc[:z_vars] = z_refs
        end

        return model, z_refs, θ, misc
    end

    return builder
end
