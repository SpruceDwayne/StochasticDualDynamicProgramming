# Example Problem: Order Execution with Decision-Dependent HFT Markup

## Overview

An investor must purchase **100 shares** of a stock over a finite trading horizon consisting of **five stages**

t = 0,1,2,3,4

At each stage the investor chooses how many shares to buy.

Large orders may trigger **high-frequency trader (HFT) front-running**, which increases the execution price through a **random markup**. The distribution of this markup **depends on the order size**, creating **decision-dependent uncertainty (DDU)**.

The objective is to **minimize expected total execution cost**.

---

# Horizon

Stages: t = 0,1,2,3,4  
Terminal stage: t = 5

The terminal value function is

Q₅ ≡ 0

---

# State Variables

The system state at stage t is

xₜ = (xₜᴾ, xₜᴼ)

where

xₜᴾ ∈ {0,…,100}  
Number of shares remaining to purchase.

xₜᴼ ∈ {0,…,100}  
Number of shares ordered in the previous period.

This is included because the **markup from period t−1 is realized at stage t**.

---

# Initial State

x₋₁ᴾ = 100  
x₋₁ᴼ = 0

---

# Decision Variable

At each stage t the investor chooses

xₜᴼ ∈ {0,…,xₜ₋₁ᴾ}

which is the number of shares to buy at stage t.

---

# State Transition

Remaining shares evolve according to

xₜᴾ = xₜ₋₁ᴾ − xₜᴼ

---

# Final Execution Constraint

All shares must be purchased by stage 3:

x₃ᴼ = x₂ᴾ  
x₃ᴾ = 0

Thus

- stages **0–3** contain trading decisions  
- stage **4** only realizes the markup cost from the stage-3 order

---

# Uncertainty

The random variables are

ξₜ = (bₜ, mₜ)

where

bₜ = base market price  
mₜ = HFT markup

---

# Base Price Process

At stage 0 the price is deterministic

b₀ = 1

For stages t = 1,…,4

bₜ ~ Uniform{0.98, 0.99, 1.00, 1.01, 1.02}

---

# HFT Markup

The markup support is

Ξₘ = {0, 0.002, 0.008}

The markup distribution depends on the order size.

---

# Decision-Dependent Regions

Orders are partitioned into four regions

I₁ = [0,15]  
I₂ = [16,40]  
I₃ = [41,75]  
I₄ = [76,100]

Define

d(xₜᴼ) ∈ {1,2,3,4}

as the region containing the order size.

Binary variables enforce region activation

δₜᵈ = 1 if xₜᴼ ∈ I_d  
Σ_d δₜᵈ = 1  
δₜᵈ ∈ {0,1}

---

# Region-Dependent Markup Distributions

Each region corresponds to a markup distribution

| Region | P(0) | P(0.002) | P(0.008) |
|------|------|------|------|
| μ₁ | 0.80 | 0.18 | 0.02 |
| μ₂ | 0.50 | 0.45 | 0.05 |
| μ₃ | 0.30 | 0.50 | 0.20 |
| μ₄ | 0.10 | 0.40 | 0.50 |

Thus

mₜ ~ μ_d   where d = d(xₜᴼ)

---

# Stage Cost

If xₜᴼ shares are purchased at stage t, the execution price per share becomes

bₜ + mₜ₊₁

The cost realized at stage t is

fₜ(xₜ, ξₜ) =
xₜᴼ · bₜ
+
xₜ₋₁ᴼ · mₜ

Interpretation

- the current order pays the base price
- the previous order pays the realized markup

---

# Value Function

The stage-t value function is

Qₜ(xₜ₋₁, ξₜ) =
min_{xₜᴼ, δₜ}
    fₜ(xₜ, ξₜ)
    + Σ_d δₜᵈ Qₜ₊₁ᵈ(xₜ)

subject to

0 ≤ xₜᴼ ≤ xₜ₋₁ᴾ

xₜᴾ = xₜ₋₁ᴾ − xₜᴼ

δₜᵈ = 1 if xₜᴼ ∈ I_d

Σ_d δₜᵈ = 1

δₜᵈ ∈ {0,1}

---

# Regional Recourse Functions

For each region

Qₜ₊₁ᵈ(xₜ) =
E_{ξₜ₊₁ᵈ}[ Qₜ₊₁(xₜ, ξₜ₊₁ᵈ) ]

where the markup distribution is

mₜ₊₁ ~ μ_d

---

# SDDiP Approximation

The algorithm maintains **region-specific value function approximations**

V̄ₜ₊₁ᵈ(xₜ)

Cuts are added in the form

θₜ ≥ vₜ^{d,i} + βₜ^{d,i} xₜ − Mₜᵈ (1 − δₜᵈ)

which activates the cut only when region d is selected.

---

# Forward Pass

At each stage solve

min_{xₜᴼ, δₜ, θₜ}

    fₜ(xₜ, ξₜ)
    + θₜ

subject to

- feasibility constraints
- region activation constraints
- all accumulated cuts

The optimal solution determines the active region

d

Then sample next uncertainty from the corresponding distribution

ξₜ₊₁ ~ μ_d

---

# Backward Pass

For each visited trial state

xₜⁱ

in region

dⁱ

solve the region-specific backward problem and generate the cut

θₜ ≥ vₜ^{d,i} + βₜ^{d,i} xₜ − Mₜᵈ (1 − δₜᵈ)

Cuts are therefore **shared only among decisions that activate the same region**.

---

# Key Modeling Properties

This example satisfies the assumptions required for DDU-SDDiP:

- finite state space
- finite region partition
- unique region activation
- finite conditional uncertainty support
- mixed-integer formulation of region membership
- bounded feasible sets

Therefore the DDU-SDDiP algorithm **converges finitely for the discretized model**.