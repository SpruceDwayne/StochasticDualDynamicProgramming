# Decision Dependent Uncertainty Extension for SDDP / SDDiP

## Purpose

This document specifies a theoretical and implementation-oriented framework for adding **decision dependent uncertainty (DDU)** to an existing Julia package for SDDP / SDDiP / risk-averse multistage stochastic programming.

The intended use of this file is as **context for a coding agent**. It should help the agent understand:

- the mathematical model
- the assumptions required for valid cut generation
- the algorithmic changes needed in forward and backward passes
- the expected internal architecture for a Julia implementation

The package already supports SDDP-style methods and relaxations of standard assumptions, including SDDiP and risk aversion. This extension adds a discretized form of decision dependent uncertainty.

---

## Core Idea

In standard SDDP, the law of future uncertainty is exogenous. In this extension, the distribution of future uncertainty depends on the current decision.

Let the decision dependent uncertainty mapping at stage `t` be

\[
g_t : \mathcal{X}_t \to \mathcal{P}_2(\mathbb{R}^d)
\]

where

\[
\mathcal{P}_2(\mathbb{R}^d)
=
\left\{
\mu \in \mathcal{P}(\mathbb{R}^d)
:
\int_{\mathbb{R}^d} \|x\|^2 \, d\mu(x) < \infty
\right\}
\]

denotes the set of probability measures on `\mathbb{R}^d` with finite second moment.

To make this tractable inside an SDDP / SDDiP framework, discretize the decision space into a finite partition.

---

## Decision Space Partition

For each stage `t`, partition the feasible decision space into disjoint subsets

\[
\mathcal{X}_t^d,\quad d \in D_t
\]

such that

\[
\bigcup_{d \in D_t} \mathcal{X}_t^d = \mathcal{X}_t
\]

and

\[
\mathcal{X}_t^d \cap \mathcal{X}_t^{d'} = \emptyset
\qquad \text{for } d \neq d'
\]

Each region `d` is associated with a fixed probability measure `\mu_d`.

Define the discretized mapping

\[
G_t(x_t, y_t)
=
\sum_{d \in D_t} \mathbf{1}_{(x_t,y_t)\in\mathcal{X}_t^d}\,\mu_d
\]

That is, if `(x_t, y_t)` falls in region `d`, then future uncertainty is sampled from `\mu_d`.

---

## Region-Dependent Dynamic Programming Formulation

For a stage problem with decision dependent uncertainty, define

\[
Q_t^d(x_{t-1}, \xi_t^d)
:=
\min_{x_t,y_t \in \mathcal{X}(x_{t-1},\xi_t^d)}
f_t(x_t,y_t,\xi_t^d)
+
\sum_{s \in D_t} \mathbf{1}_s \,\mathcal{Q}_{t+1}^s(x_t)
\]

subject to

\[
\mathbf{1}_d = 1 \iff (x_t,y_t) \in \mathcal{X}_t^d
\qquad \forall d \in D_t
\]

\[
\mathbf{1}_d \in \{0,1\}
\qquad \forall d \in D_t
\]

where

\[
\mathcal{Q}_{t+1}^d(x_t)
=
\mathbb{E}^{d}\left[Q_{t+1}^d(x_t,\xi_{t+1})\right]
\]

and the expectation is taken under the distribution `\mu_d`.

The key difference from classical SDDP is that there is no longer a single recourse function at stage `t+1`. Instead there is a **family of recourse functions**, one for each region.

---

## Region-Specific Value Function Approximations

The algorithm must maintain lower approximations

\[
\underline{\mathcal{V}}_t^d(x_{t-1})
\]

for each stage `t` and each region `d \in D_t`.

So instead of a single approximation per stage, we maintain

\[
\left\{
\underline{\mathcal{V}}_t^d
\right\}_{d \in D_t}
\]

Cuts are region-specific.

---

## Cut Structure

Cuts should activate only when the corresponding region is selected.

The proposed cut form is

\[
\theta_t
\ge
v_t^{d,i}
+
(\beta_t^{d,i})^\top x_t
+
M(\mathbf{1}_d - 1)
\]

where

- `v_t^{d,i}` is the cut intercept
- `\beta_t^{d,i}` is the cut slope / subgradient
- `M` is a sufficiently large constant
- `\mathbf{1}_d` is the binary activation variable for region `d`

Interpretation:

- if `\mathbf{1}_d = 1`, the cut is active and becomes
  \[
  \theta_t \ge v_t^{d,i} + (\beta_t^{d,i})^\top x_t
  \]
- if `\mathbf{1}_d = 0`, the cut is relaxed by `-M`

Thus cuts switch on and off depending on which future distribution the current decision implies.

---

## Why Benders Cuts Are Not Enough

In this setup, each subproblem contains integer decisions because:

- region activation uses binary variables
- the broader package already includes SDDiP-style integrality
- future region selection can also depend on mixed-integer structure

Hence classical Benders cuts from standard convex SDDP are generally not sufficient.

Instead, if the assumptions below hold, **Lagrangian cuts** from SDDiP can be used to obtain cuts that are tight and valid within each region.

---

## Assumptions

### A12 — Binarity of states

For each node `n`, the state variable satisfies

\[
x_{a(n)} \in \{0,1\}^{m}
\]

for some finite `m`.

This is the same structural requirement used in SDDiP to ensure tightness of Lagrangian cuts.

---

### A13 — Exact partition of the decision space

For each stage `t`, there exists a finite index set `D_t` and pairwise disjoint sets

\[
\{\mathcal{X}_t^d\}_{d \in D_t}
\]

such that

\[
\bigcup_{d \in D_t} \mathcal{X}_t^d = \mathcal{X}_t,
\qquad
\mathcal{X}_t^d \cap \mathcal{X}_t^{d'} = \emptyset
\quad \text{for } d \neq d'
\]

and

\[
g_t(x_t,y_t) = \mu_d
\qquad \text{whenever } (x_t,y_t) \in \mathcal{X}_t^d
\]

This means the discretized mapping is exact on each partition cell.

---

### A14 — Unique region activation

For each stage `t`, exactly one region indicator is active:

\[
\sum_{d \in D_t} \mathbf{1}_d = 1
\qquad
\mathbf{1}_d \in \{0,1\}
\quad \forall d \in D_t
\]

and the model enforces

\[
\mathbf{1}_d = 1
\iff
(x_t,y_t) \in \mathcal{X}_t^d
\]

This ensures a unique conditional distribution is selected at each stage.

---

### A15 — Boundedness

For each node `n`, the feasible set of `(x_n, y_n)` is compact.

In particular:

- the state space is bounded
- the decision space is bounded

This is needed to ensure that the big-`M` cuts remain globally valid.

---

### A16 — Finite support of conditional distributions

For each `d \in D_t`, the probability measure `\mu_d` has finite support.

This allows conditional expectations to be written as finite probability-weighted sums.

---

### A17 — Convex recourse within regions

For each `d \in D_t`, the recourse function

\[
\mathcal{Q}_{t+1}^d(x_t)
\]

is convex in `x_t` when the integrality of future-stage variables, except state variables, is relaxed.

This is the regionwise analogue of the convexity assumption used in classical SDDP.

It is essential for the validity of region-specific supporting hyperplanes / Lagrangian cuts.

---

## Remark on Convex Recourse Within Regions

Assumption A17 ensures that for each region `d`, the relaxed recourse function `\mathcal{Q}_{t+1}^d(x_t)` is convex in the state `x_t`.

This matters because the cuts

\[
\theta_t
\ge
v_t^{d,i}
+
(\beta_t^{d,i})^\top x_t
+
M(\mathbf{1}_d - 1)
\]

must underestimate the recourse function everywhere when region `d` is active.

Convexity guarantees that any supporting hyperplane constructed at a trial point `x_t^i` yields a valid lower bound throughout the region.

Without this property, a cut might overestimate the recourse at some points, destroying validity.

---

## Interpretation of the Assumptions

Assumptions A12–A16 are natural extensions of the assumptions already used in SDDP / SDDiP.

- A12 formalizes the binarity requirement used in SDDiP to ensure tight Lagrangian cuts.
- A13 and A14 formalize the discretization of the uncertainty mapping and guarantee that exactly one distribution is active.
- A15 and A16 provide boundedness and finite support, which are needed for cut finiteness and computable expectations.
- A17 is the key structural extension: convexity is now required **within each conditional region**, not globally under a single distribution.

---

## Cut Properties Needed for Convergence

Cuts must satisfy:

1. validity
2. tightness
3. finiteness

The implementation should be designed so these properties are preserved regionwise.

---

## Validity of Region-Specific Cuts

Fix a region `d \in D_t`.

By Assumption A17, the relaxed recourse function `\mathcal{Q}_{t+1}^d(x_t)` is convex in `x_t`.

Therefore any supporting hyperplane built at a trial point `x_t^i` satisfies

\[
\mathcal{Q}_{t+1}^d(x_t)
\ge
v_t^{d,i} + (\beta_t^{d,i})^\top x_t
\qquad \forall x_t
\]

If `\mathbf{1}_d = 1`, the cut

\[
\theta_t
\ge
v_t^{d,i}
+
(\beta_t^{d,i})^\top x_t
+
M(\mathbf{1}_d - 1)
\]

reduces to

\[
\theta_t \ge v_t^{d,i} + (\beta_t^{d,i})^\top x_t
\]

and is therefore valid.

If `\mathbf{1}_d = 0`, the right-hand side becomes

\[
v_t^{d,i} + (\beta_t^{d,i})^\top x_t - M
\]

Since the feasible set is bounded by A15, `M` can be chosen large enough so that this remains a valid lower bound everywhere.

Hence the cut is globally valid.

---

## Tightness of Region-Specific Cuts

Cuts are generated from Lagrangian dual multipliers at the current forward trial point `x_t^i`.

By the tightness property of Lagrangian cuts in SDDiP:

\[
v_t^{d,i}
=
\underline{\mathcal{Q}}_{t+1}^d(x_t^i)
\]

and `\beta_t^{d,i}` is a valid subgradient at `x_t^i`.

Thus, when evaluated at the generating point with `\mathbf{1}_d = 1`, the cut satisfies equality:

\[
\theta_t
=
\underline{\mathcal{Q}}_{t+1}^d(x_t^i)
\]

So the cut is tight at its generating point.

---

## Finiteness of Region-Specific Cuts

By A12, A15, and A16:

- states are binary
- feasible sets are bounded
- conditional distributions have finite support

Therefore each region-specific subproblem admits only finitely many distinct dual basic solutions.

Since cuts are generated from such solutions, only finitely many distinct cuts can arise within each region.

Because `D_t` is finite, the total number of possible cuts at stage `t` is also finite.

---

## High-Level Algorithmic Change

The main algorithmic change is that the package must maintain a **family of value function approximations** at each stage rather than a single one.

At iteration `i`, assume approximations

\[
\underline{\mathcal{V}}_t^{d,i}
\]

exist for all stages `t` and regions `d`.

---

## Forward Pass

The forward pass proceeds stagewise.

At stage `t`, given state `x_{t-1}^{i,k}`, solve

\[
\min_{x_t, y_t, \mathbf{1}_d}
f_t(x_t,y_t,\xi_t)
+
\theta_t
\]

subject to:

- the stage feasibility constraints
- the region membership / activation constraints
- all currently available active-region cuts
- exactly one region active

This solution determines:

- the next state `x_t^{i,k}`
- the active region `d^{i,k}`

Then sample next-stage uncertainty conditionally on the active region:

\[
\xi_{t+1}^{i,k} \sim \mu_{d^{i,k}}
\]

Thus scenario generation becomes endogenous: the distribution used at stage `t+1` depends on the decision taken at stage `t`.

### Forward pass implementation requirement

The forward pass sampler must not sample from a fixed stagewise-independent law. It must sample from the distribution associated with the chosen region.

---

## Backward Pass

In the backward pass, proceed regionwise.

For each visited trial point `(x_t^{i,k}, d^{i,k})`, solve the corresponding region-specific subproblems for each scenario in the support of `\mu_{d^{i,k}}`:

\[
\underline{Q}_{t+1}^{d^{i,k}}(x_t^{i,k},\xi_{t+1,j}^{d^{i,k}})
=
\min
\left\{
f_{t+1}(x_{t+1},y_{t+1},\xi_{t+1,j}^{d^{i,k}})
+
\sum_{s \in D_{t+1}}
\mathbf{1}_s(x_{t+1},y_{t+1})\,
\underline{\mathcal{V}}_{t+2}^s(x_{t+1})
\right\}
\]

Using the Lagrangian multipliers from these solves, construct a cut of the form

\[
\theta_t
\ge
v_t^{d^{i,k},i}
+
(\beta_t^{d^{i,k},i})^\top x_t
+
M(\mathbf{1}_{d^{i,k}} - 1)
\]

and add it only to the approximation associated with region `d^{i,k}`.

### Backward pass implementation requirement

Cuts are **not shared globally across all regions**. They belong to the region under which they were generated.

---

## Data Model Expectations for a Julia Implementation

The coding agent should structure the implementation around the following concepts.

### 1. Region definition

A region should represent:

- a region identifier
- the membership logic or constraints defining `\mathcal{X}_t^d`
- the conditional distribution `\mu_d`

Possible conceptual structure:

```julia
struct DDURegion
    id::Int
    # user-defined metadata describing the region
    # distribution object or explicit support/probabilities
end