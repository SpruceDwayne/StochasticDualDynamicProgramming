using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "SDDPBAPE"))
Pkg.instantiate()

using SDDPBAPE

Pkg.add("Revise")