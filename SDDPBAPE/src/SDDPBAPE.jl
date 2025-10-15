module SDDPBAPE

using JuMP
using LinearAlgebra
# If your ω types use SparseMatrixCSC, also:
# using SparseArrays

include("SDDP.jl")      # defines: Cut, ValueFn, Stage, SDDP, add_cut!, evaluate, run_sddp
include("passes.jl")    # defines: ForwardRecord, forward_pass!, backward_pass!, compute_cut!
include("deterministic_checks.jl") #defines solve_extensive_control for validating SDDP outputs

# Example helper (optional)
collect_samples(st::Stage; N::Int=3) = (st.sampler() for _ in 1:N)

# ---- Exports ----
export Cut, ValueFn, Stage, SDDP,
       evaluate, add_cut!,
       forward_pass_online!, backward_pass_expected!,
       forward_pass!, ForwardRecord, compute_cut!,   
       BaseStageData, OmegaRef, OmegaStageData,
       collect_samples,solve_extensive_control,run_sddp!

end # module
