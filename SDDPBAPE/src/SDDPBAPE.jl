module SDDPBAPE

using JuMP
using LinearAlgebra
# If your ω types use SparseMatrixCSC, also:
# using SparseArrays

#definitions
include("SDDP.jl")                   # Cut, ValueFn, Stage, SDDP, MarkovSDDP, add_cut!, evaluate, run_sddp!, run_markov_sddp!,get_V!x
include("passes.jl")                 # ForwardRecord, forward_pass!, forward_pass_online!, forward_pass_markov_online!,
                                     # backward_pass_expected!, backward_pass_markov_expected!, compute_cut!
include("deterministic_checks.jl")   # solve_extensive_control for validating SDDP outputs


# Example helper (optional)
collect_samples(st::Stage; N::Int=3) = (st.sampler() for _ in 1:N)

# ---- Exports ----
export Cut, ValueFn, Stage, SDDP, MarkovSDDP,
       evaluate, add_cut!,get_V!,
       # IID / standard SDDP passes
       forward_pass_online!, backward_pass_expected!,
       forward_pass!, ForwardRecord, compute_cut!,
       # Markov SDDP passes
       forward_pass_markov_online!, backward_pass_markov_expected!,
       #Risk averse SDDP passes
       _avar_dual_weights,backward_pass_markov_rho!,
       # Data containers
       BaseStageData, OmegaRef, OmegaStageData,
       # Utilities
       collect_samples, solve_extensive_control,
       # Drivers
       run_sddp!, run_markov_sddp!,run_markov_sddp_rho!

end # module

