module GurobiExt

using SDDPBAPE
using Gurobi

# This extension is intentionally minimal.
# Its presence tells Julia's package manager that SDDPBAPE supports Gurobi
# as an optional dependency when it is installed by the user.
#
# No code is required here because SDDPBAPE is already solver-agnostic:
# - Stage subproblem models are built by user-supplied `build` functions;
#   pass `Model(Gurobi.Optimizer)` there directly.
# - `LevelMethodConfig` accepts any MOI-compatible optimizer via its
#   `optimizer` field.
# - `solve_extensive_control` accepts an `optimizer` keyword argument.
#
# See README.md for usage examples.

end # module GurobiExt
