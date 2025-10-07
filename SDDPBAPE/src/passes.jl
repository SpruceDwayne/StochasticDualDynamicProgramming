

"Forward pass: sample one path and return visited trial states x̄_t."
function forward_pass!(algo::SDDP)
    xbars = Vector{Vector{Float64}}(undef, algo.T)
    vfnext = nothing
    for t in 1:algo.T
        st = algo.stages[t]
        ω = st.sampler()
        (model, x_t, θ, misc) = st.build(t, t < algo.T ? algo.V[t+1] : ValueFn{Float64}(), ω)

        # Solve (user chooses solver in build)
        optimize!(model)

        # Record the visited state (value of x_t)
        xbars[t] = value.(x_t)
    end
    # store for backward pass
    for t in 1:algo.T
        push!(algo.trial_points[t], xbars[t])
    end
    return xbars
end

"Backward pass: for t = T..1, build a cut at each stored x̄_t and add to V_t."
function backward_pass!(algo::SDDP; batch_size::Int=1)
    for t in algo.T:-1:1
        st = algo.stages[t]
        vfnext = (t < algo.T) ? algo.V[t+1] : ValueFn{Float64}() # empty V_{T+1} = 0
        # process a small batch of most recent trial points (you can schedule differently)
        for xbar in Iterators.take(Iterators.reverse(algo.trial_points[t]), batch_size)
            # Build and solve several noise realizations at fixed x_t = x̄_t
            Esupport = 0.0
            βbar = zeros(Float64, st.state_dim)  # expected subgradient wrt x_t
            nS = 0
            for ω in collect_samples(st)  # you define this: could be full support or Monte Carlo subsample
                (model, x_t, θ, misc) = st.build(t, vfnext, ω; fix_state = xbar)

                # Solve
                optimize!(model)

                # Read optimal next state (if needed for chaining) and which cut is active for θ
                θval = value(θ)
                # We need the subgradient wrt x_t. Two common routes:
                # (A) If your build() exposes A_t so that x_{t+1} = A_t * x_t + ..., and you know β_{t+1} (active),
                #     use β_x_t = A_t' * β_{t+1}.
                # (B) If you enforce x_t via equality constraints, get duals on those eqs directly.

                β_xt = misc[:β_xt]  # provide this from build() for clarity
                Esupport += st.ϕ(ω) * (objective_value(model))  # immediate + approx future
                βbar    .+= st.ϕ(ω) * β_xt
                nS += 1
            end
            # Cut: α + β' x ≤ V_t(x). Make it tight at x = x̄_t with value Esupport.
            α = Esupport - dot(βbar, xbar)
            add_cut!(algo.V[t], α, βbar, t)
        end
    end
end




