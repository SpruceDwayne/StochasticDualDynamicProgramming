module SDDPBAPE

using JuMP
using LinearAlgebra
# If your ω types use SparseMatrixCSC, also:
# using SparseArrays

#definitions
include("SDDP.jl")                   # Cut, ValueFn, Stage, SDDP, MarkovSDDP, add_cut!, evaluate, run_sddp!, run_markov_sddp!, get_V!
include("passes.jl")                 # ForwardRecord, forward_pass!, forward_pass_online!, forward_pass_markov_online!,
                                     # backward_pass_expected!, backward_pass_markov_expected!, compute_cut!
include("deterministic_checks.jl")   # solve_extensive_control for validating SDDP outputs
include("SDDiP.jl")                  # SDDiPConfig, solve_lagrangian_dual!, compute_sddip_cut!,
                                     # backward_pass_sddip!, backward_pass_markov_sddip!,
                                     # run_sddip!, run_markov_sddip!, binarize
include("DDU.jl")                    # DDURegion, DDUModelCache, DDUSDDP, DDUForwardRecord,
                                     # get_V_ddu!, get_or_build_ddu_model!,
                                     # forward_pass_ddu_online!, backward_pass_ddu_sddip!,
                                     # compute_ddu_lb!, run_ddu_sddip!


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
       # Drivers (SDDP)
       run_sddp!, run_markov_sddp!, run_markov_sddp_rho!,
       # SDDiP
       LevelMethodConfig, SDDiPConfig, solve_lagrangian_dual!, solve_lagrangian_dual_level!,
       compute_sddip_cut!, backward_pass_sddip!, backward_pass_markov_sddip!,
       run_sddip!, run_markov_sddip!, binarize,
       # DDU
       DDURegion, DDUModelCache, DDUSDDP, DDUForwardRecord,
       get_V_ddu!, get_or_build_ddu_model!,
       forward_pass_ddu_online!, backward_pass_ddu_sddip!,
       compute_ddu_lb!, run_ddu_sddip!, _read_active_region

end # module

