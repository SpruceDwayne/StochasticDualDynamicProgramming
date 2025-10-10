module SDDPBAPE

using JuMP
using LinearAlgebra
# If your ω types use SparseMatrixCSC, also:
# using SparseArrays

include("SDDP.jl")      # defines: Cut, ValueFn, Stage, SDDP, add_cut!, evaluate
include("passes.jl")    # defines: ForwardRecord, forward_pass!, backward_pass!, compute_cut!

# Example helper (optional)
collect_samples(st::Stage; N::Int=3) = (st.sampler() for _ in 1:N)

# ---- Exports ----
export Cut, ValueFn, Stage, SDDP,
       evaluate, add_cut!,
       forward_pass!, backward_pass!, collect_samples,
       # passes extras (export if you want them public)
       ForwardRecord, compute_cut!,
       # scenario payloads / tree API (if defined in SDDP.jl)
       BaseStageData, OmegaRef, OmegaStageData,
       TreeNode, ScenarioTree, Trajectory,
       enumerate_trajectories, sample_trajectories

end # module

