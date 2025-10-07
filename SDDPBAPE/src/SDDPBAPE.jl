module SDDPBAPE

using JuMP
using LinearAlgebra

include("SDDP.jl")      # brings Cut, ValueFn, Stage, SDDP, add_cut!, evaluate
include("passes.jl")    # forward_pass!, backward_pass!

# simple sampling helper used by passes.jl (tweak as you like)
collect_samples(st::Stage; N::Int=3) = (st.sampler() for _ in 1:N)

export Cut, ValueFn, Stage, SDDP,
       evaluate, add_cut!,
       forward_pass!, backward_pass!, collect_samples

end # module
