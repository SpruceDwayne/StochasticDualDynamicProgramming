# StochasticDualDynamicProgramming
This is a repo where I will store code used for my master thesis and the accompanying Thesis Preparation Project both focusing on SDDP and it's extensions including a new extension for Decidsion Dependent Uncertainty (DDU). Some of the examples in the examples folder are quite large as they were in fact used for application chapters in my projects and not made as small examples for documenting the package.

Blabla with other useful infos

Decide if .md files are useful as explanations and context about the underlying theory or should be removed.

## Solver configuration

The package uses [HiGHS](https://github.com/jump-dev/HiGHS.jl) by default. Any JuMP-compatible solver can be substituted.

### Stage subproblem models

Stage subproblem models are built by your own `build` function, so passing a different solver is as simple as constructing the model with a different optimizer:

```julia
using Gurobi

function my_build(t, vf_next, ω; fix_state)
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    # ... add variables, constraints, objective, θ epigraph variable ...
    return model, x_state, θ, Dict()
end
```

### Level-method Lagrangian dual (SDDiP)

`LevelMethodConfig` takes any optimizer that supports quadratic objectives:

```julia
using Gurobi
cfg = SDDiPConfig(
    cut_type  = :lagrangian,
    level_cfg = LevelMethodConfig(optimizer = Gurobi.Optimizer),
)
```

### Extensive-form validation

`solve_extensive_control` accepts an `optimizer` keyword (defaults to `HiGHS.Optimizer`):

```julia
using Gurobi
x0, obj, ef = solve_extensive_control(m;
    B0        = [100.0],
    c0        = [1.06],
    optimizer = Gurobi.Optimizer,
)
```
