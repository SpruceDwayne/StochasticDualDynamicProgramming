using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
Pkg.instantiate()

using SDDPBAPE

using SparseArrays
using LinearAlgebra

# Base data
n, m = 2, 2
A = sparse(1.0I, n, n)              # Float64 identity
B = sparse(1.0I, n, n)              # Float64 identity
G = sparse([1.0 1.0; -1.0 0.0])     # already Float64

base = BaseStageData(A, B, G)       # now all eltype == Float64 → works

# Two second-stage nodes differ by price and demand
ω1 = OmegaRef(; base, c_u = [1.0, 2.0], c_x = zeros(n), d = zeros(n), h = [3.0, 0.0])
ω2 = OmegaRef(; base, c_u = [2.0, 1.0], c_x = zeros(n), d = zeros(n), h = [2.5, 0.0])

lvl1 = [TreeNode(1, nothing, OmegaRef(base, [1.0,1.0], zeros(n), zeros(n), [2.0,0.0]), 1.0)]
lvl2 = [TreeNode(2, 1, ω1, 0.6),
        TreeNode(3, 1, ω2, 0.4)]

tree = ScenarioTree([lvl1, lvl2])

# Sample a trajectory online (per-stage conditional sampling)
traj = sample_trajectories(tree, 10)  # IID under the tree measure

# Or enumerate and sample IID from the path distribution
all_paths = enumerate_trajectories(tree)



