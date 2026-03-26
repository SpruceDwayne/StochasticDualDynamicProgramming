# Level Method for Lagrangian Dual Optimization

## Overview

The **level method** is a first-order optimization algorithm designed for solving **concave, possibly non-smooth dual problems**, such as those arising from **Lagrangian relaxation**.

It is particularly effective when:
- The dual function is **non-differentiable**
- Subgradients are available but noisy
- Subgradient methods converge too slowly

The method can be seen as a **stabilized cutting-plane / bundle method**, combining:
- Memory of past subgradients (cuts)
- A projection step for stability
- A target “level” to guide progress

---

## Problem Setup

Consider a constrained minimization problem:

    minimize    f(x)
    subject to  h(x) ≤ 0

The Lagrangian is:

    L(x, λ) = f(x) + λᵀ h(x),     λ ≥ 0

The dual function is:

    q(λ) = inf_x L(x, λ)

The dual problem is:

    maximize    q(λ)
    subject to  λ ≥ 0

### Key properties

- q(λ) is **concave**
- q(λ) is often **non-smooth**
- A subgradient is:

    g(λ) = h(x_λ)

where x_λ minimizes L(x, λ)

---

## Core Idea

Instead of using only the current subgradient (like in subgradient ascent), the level method:

1. Stores past evaluations (cuts)
2. Builds a piecewise-linear model of the dual
3. Maintains lower and upper bounds
4. Chooses a target **level** between them
5. Moves to the closest point that satisfies this level

This avoids oscillations and improves convergence stability.

---

## Cutting-Plane Model

At iteration k, we have points λ¹, ..., λᵏ with:
- dual values q(λʲ)
- subgradients gʲ

Define:

    m_k(λ) = min_j { q(λʲ) + gʲᵀ(λ - λʲ) }

Because q is concave:

    q(λ) ≤ q(λʲ) + gʲᵀ(λ - λʲ)

So m_k(λ) is an **upper approximation** of q(λ):

    q(λ) ≤ m_k(λ)

---

## Bounds

We maintain:

Lower bound (best known feasible value):

    LB = max_j q(λʲ)

Upper bound (from model):

    UB = max_{λ ∈ Λ} m_k(λ)

where Λ is the feasible set (typically λ ≥ 0, possibly bounded).

---

## Level Definition

Choose parameter:

    α ∈ (0, 1)

Define:

    ℓ = α * LB + (1 - α) * UB

This is the **target level**.

---

## Projection Step

Compute next iterate:

    λ⁺ = argmin_{λ ∈ Λ}  (1/2) ||λ - center||²
          subject to     m_k(λ) ≥ ℓ

Interpretation:
- Stay close to current point (or center)
- Move into region where model predicts sufficient improvement

---

## Algorithm
1. Build model:
       m_k(λ) = min_j { q(λʲ) + gʲᵀ(λ - λʲ) }

2. Compute:
       UB = max_{λ ∈ Λ} m_k(λ)

3. Stop if:
       UB - LB ≤ ε

4. Set level:
       ℓ = α * LB + (1 - α) * UB

5. Projection step:
       λ^{k+1} = argmin_{λ ∈ Λ} 1/2 ||λ - center||²
                  s.t. m_k(λ) ≥ ℓ

6. Evaluate:
       q(λ^{k+1})
       g^{k+1}

7. Update:
       LB = max(LB, q(λ^{k+1}))

8. Store new cut


---

## Practical Notes

### Subgradients

For Lagrangian duals:

    g(λ) = h(x_λ)

This comes directly from the primal subproblem.

---

### Choice of Λ (feasible set)

To ensure bounded subproblems, one typically uses:
- λ ≥ 0 with upper bounds
- box constraints
- trust region constraints

---

### Center choice

Common options:
- current iterate
- best iterate so far
- running average

---

### Memory management

To avoid excessive growth:
- keep only recent cuts
- drop inactive cuts
- use aggregation

---

## Why It Works Well

Compared to subgradient ascent:

- Uses **all past information**, not just current gradient
- Avoids **zig-zagging**
- No need for delicate step-size tuning
- More stable progress

---

## Computational Tradeoff

Each iteration is more expensive:
- solve projection problem
- maintain model

But:
- **far fewer iterations**
- much better practical convergence

---

## When to Use

Best suited for:

- Lagrangian duals
- Non-smooth convex optimization
- Large-scale decomposition methods
- Integer programming relaxations

---

## Summary

The level method is a stabilized optimization technique for non-smooth dual problems. It builds a cutting-plane model of the dual, maintains bounds, and computes iterates via projection onto a level set that guarantees sufficient improvement. It is typically much more reliable and faster than plain subgradient methods in practice.