# SDDiP Implementation Guide

## Core Concept
SDDiP extends SDDP to handle **binary state variables** through a key reformulation and Lagrangian cuts.

## Key Reformulation (Section 2)
Standard MSIP subproblem becomes:
```
Qₙ(x_{a(n)}) = min fₙ(xₙ, yₙ) + Σ qₙₘQₘ(xₙ)
                s.t. (zₙ, xₙ, yₙ) ∈ Xₙ
                     zₙ = x_{a(n)}        # Copy constraint
                     zₙ ∈ [0,1]ᵈ
                     xₙ ∈ {0,1}ᵈ         # Binary states
```
**Critical**: Introduce auxiliary variable `zₙ` (continuous) that copies parent state `x_{a(n)}` (binary).

## Cut Types (Section 4)

### 1. Strengthened Benders' Cut (Recommended - Section 4.4)
- Solve LP relaxation to get dual multiplier πₘ for constraint zₘ = xₙ
- Solve MIP: Lₘ(πₘ) = min{fₘ(xₘ,yₘ) + θₘ - πₘᵀzₘ : (zₘ,xₘ,yₘ,θₘ) ∈ X''ₘ}
- Cut: θₙ ≥ Σ qₙₘ[Lₘ(πₘ) + πₘᵀxₙ]
- **Valid & finite, not tight but computationally efficient**

### 2. Lagrangian Cut (Tight - Section 4.3)
- Solve Lagrangian dual: max_π {Lₘ(π) + πᵀx_{a(m)}}
- Use subgradient method with tolerance 10⁻⁴
- Cut coefficients: (vₘ, πₘ) where vₘ = Lₘ(πₘ)
- **Valid, tight & finite - guarantees convergence**

### 3. Integer Optimality Cut (Section 4.2)
- Solve child problems to optimality: vₘ = Qₘ(xₙ)
- Cut: θₙ ≥ (v̄ₙ - Lₙ)[Σ(xₙⱼ-1)xₙⱼ + Σ(xₙⱼ-1)xₙⱼ] + v̄ₙ
  where v̄ₙ = Σ qₙₘvₘ
- **Tight only at evaluated point, slow convergence**

## Implementation Strategy (Section 6.4 Summary)

### Best Practice Combination: SB + I. Function below is only a sketch
```julia
function solve_lagrangian_dual(model, x_parent, tolerance=1e-4)
    # Initialize
    π = get_lp_dual(model, x_parent)  # Start from LP dual
    best_L = -Inf
    best_π = π
    step_size = 1.0
    
    for iter in 1:max_iterations
        # Solve inner MIP: L(π) = min{f(x,y) + θ - π'z : constraints}
        L_val, x_sol, z_sol = solve_lagrangian_subproblem(model, π)
        
        # Update best
        obj = L_val + π'x_parent
        if obj > best_L
            best_L = obj
            best_π = π
        end
        
        # Subgradient is (x_parent - z_sol)
        subgrad = x_parent - z_sol
        
        # Check convergence
        if norm(subgrad) < tolerance
            break
        end
        
        # Subgradient step
        π = π + step_size * subgrad
        
        # Update step size (diminishing)
        step_size *= 0.95  # or other schedule
    end
    
    return best_L, best_π
end

function solve_lagrangian_subproblem(model, π)
    # This is just a MIP!
    # min f(x,y) + θ - π'z
    # s.t. (z,x,y) ∈ X
    #      x ∈ {0,1}^d
    #      z ∈ [0,1]^d  
    #      θ ≥ existing_cuts(x)
    
    # Modify objective: add -π'z term
    # Solve with JuMP/HiGHS
    # Return objective value, x_sol, z_sol
end


function backward_pass!(model, x_forward)
    for each child node m
        # Step 1: LP relaxation for dual
        π_m = solve_lp_relaxation_dual(m, x_forward)
        
        # Step 2: Strengthened Benders
        L_m = solve_lagrangian_subproblem(m, π_m, x_forward)
        
        # Step 3: Integer optimality (optional)
        v_m = solve_mip(m, x_forward)  # Full solve
        
        # Add both cuts
        add_cut!(model, SB_cut(L_m, π_m))
        add_cut!(model, IO_cut(v_m, x_forward))
    end
end
```

### Forward Pass
- Use **1-3 sample paths** (1 often best per Table 2)
- Solve: min{fₙ(xₙ,yₙ) + ψₙ(xₙ) : (zₙ,xₙ,yₙ) ∈ Xₙ, zₙ = x_{parent}}
- Approximation ψₙ(xₙ) = min{θₙ : accumulated cuts}

## Handling General Integer/Continuous States (Section 5)

Binary expansion for integer x ∈ {0,...,U}:
```julia
κ = floor(log2(U)) + 1
x = Σ_{i=1}^κ 2^{i-1} λᵢ,  λᵢ ∈ {0,1}
```

Binary approximation for continuous x ∈ [0,U] with precision ε:
```julia
κ = floor(log2(U/ε)) + 1  
x ≈ Σ_{i=1}^κ ε·2^{i-1} λᵢ,  λᵢ ∈ {0,1}
```

Total binary vars: k ≤ d(⌊log₂(M√d/ε)⌋ + 1)

## Convergence (Theorem 2)
Requires cuts that are:
1. **Valid**: Qₙ(x) ≥ vₙ + πₙᵀx for all x ∈ {0,1}ᵈ
2. **Tight**: Q'ₙ(x_forward, ψₙ₊₁) = vₙ + πₙᵀx_forward
3. **Finite**: Only finitely many distinct cuts possible

SB+I satisfies valid & finite. Adding Lagrangian cuts ensures tightness.

## Practical Settings (Section 6)
- MIP tolerance: 10⁻⁴ (relax to 0.05 for hard problems)
- Lagrangian dual tolerance: 10⁻⁴
- Forward samples: 1-3 paths

## Key Differences from SDDP
1. **States must be binary** (or binarized)
2. **Copy constraint** zₙ = x_{parent} is crucial
3. **MIP solves** in backward pass (not just LP)
4. **Multiple cut types** improve convergence