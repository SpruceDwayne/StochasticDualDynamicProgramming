# SDDiP Smoke Test: 2-Stage Binary MSIP (Toy Unit Commitment)

## Goal
Provide a minimal binary MSIP that exercises the SDDiP machinery:
- binary state variable x_t
- copy variable z_t ∈ [0,1] copying parent state through constraint z_t = x_parent
- stage-2 stochastic recourse with scenarios
- backward pass needs an LP relaxation solve (to get dual π for z = x_parent) and a MIP solve (Lagrangian subproblem / SB cut)
- optional integer-optimality cut (full MIP solve of child)

This model is intentionally tiny so a single forward/backward iteration should run fast.

---

## Sets / Stages / Scenarios
- Stages: t = 1,2
- Scenarios at stage 2: s ∈ {L, H}
- Probabilities: P(L)=0.5, P(H)=0.5

---

## State (Binary)
- x₁ ∈ {0,1}: generator ON state after stage 1 (this is the state passed to stage 2)
- x₂ ∈ {0,1}: generator ON state after stage 2 (not strictly needed beyond stage 2, but include for structure)

**SDDiP reformulation requirement (copy constraint)**
At stage 2 introduce:
- z₂ ∈ [0,1] (continuous copy of parent)
- constraint: z₂ = x₁

---

## Decisions (Binary + Continuous)
Stage 1:
- x₁ ∈ {0,1}  (state decision)
- u₁ ∈ {0,1}  (startup decision)
- g₁ ≥ 0      (generation)
- shed₁ ≥ 0   (load shed)

Stage 2 (for each scenario s):
- x₂(s) ∈ {0,1}  (state decision at stage 2)
- u₂(s) ∈ {0,1}  (startup at stage 2)
- g₂(s) ≥ 0      (generation)
- shed₂(s) ≥ 0   (load shed)
- z₂(s) ∈ [0,1]  (copy variable; could be shared across scenarios or per-scenario—either is fine for the toy)
- θ₂(s) (cost-to-go approximation variable, can be 0 or omitted since stage 2 is terminal)

---

## Parameters (simple numbers)
Capacity:
- Gmax = 10

Demands:
- d₁ = 6
- d₂(L) = 4
- d₂(H) = 9

Costs:
- startup cost: SU = 2
- fixed on cost per stage: F = 0.5
- variable generation cost: c = 1.0
- load shedding penalty: M = 100 (big)

---

## Constraints

### Stage 1 constraints
1) Power balance:
   g₁ + shed₁ = d₁

2) Capacity if on:
   0 ≤ g₁ ≤ Gmax * x₁

3) Startup logic (if you start at t=1, pay SU):
   u₁ ≥ x₁
   u₁ ∈ {0,1}

(Interpretation: starting from OFF at t=0, so u₁ is effectively x₁.)

---

### Stage 2 constraints (per scenario s)
**Copy constraint (core SDDiP piece):**
4) z₂(s) = x₁
   z₂(s) ∈ [0,1]
   x₁ ∈ {0,1}

5) Power balance:
   g₂(s) + shed₂(s) = d₂(s)

6) Capacity if on:
   0 ≤ g₂(s) ≤ Gmax * x₂(s)

7) Startup logic tied to copied parent state:
   u₂(s) ≥ x₂(s) - z₂(s)
   u₂(s) ∈ {0,1}

(This is the important place where z₂ appears in constraints, so the LP relaxation dual π for z₂ = x₁ will be meaningful.)

---

## Objective
Minimize stage 1 cost + expected stage 2 cost

Stage 1 cost:
- SU*u₁ + F*x₁ + c*g₁ + M*shed₁

Stage 2 cost in scenario s:
- SU*u₂(s) + F*x₂(s) + c*g₂(s) + M*shed₂(s)

Total objective:
min [ SU*u₁ + F*x₁ + c*g₁ + M*shed₁ ]
  + Σ_s P(s) * [ SU*u₂(s) + F*x₂(s) + c*g₂(s) + M*shed₂(s) ]

Terminal stage: no further cost-to-go needed.

---

## What the optimal policy should look like (sanity expectations)
- Because M is huge, the model will prefer turning on the generator to avoid shedding.
- At stage 1: x₁ will likely be 1 to supply demand d₁=6 without shedding.
- At stage 2:
  - If x₁=1, staying on avoids startup and supplies both scenarios cheaply.
  - If x₁=0 (forced in a test), scenario H will trigger startup or shedding.

This ensures the value function in stage 1 genuinely depends on x₁, so SDDiP cuts are not degenerate.

---

## SDDiP Algorithm Hooks (how to use this test)
Forward pass:
- Sample 1 path (either L or H)
- Solve stage 1, get x₁^k
- Solve stage 2 for sampled scenario with constraint z₂ = x₁^k

Backward pass (at stage 1):
For each scenario s:
1) Solve LP relaxation of stage-2 subproblem (relax x₂(s), u₂(s) ∈ [0,1]) to obtain dual π_s
   associated with the copy constraint z₂(s) = x₁.

2) Strengthened Benders cut (SB):
   - Solve MIP: L_s(π_s) = min { stage2_cost(s) - π_s * z₂(s) : stage2_constraints(s), cuts_on_θ (if any) }
   - Add cut to stage-1 approximation:
     θ₁ ≥ Σ_s P(s) * ( L_s(π_s) + π_s * x₁ )

3) Optional integer optimality cut (IO):
   - Solve full MIP of stage-2 at x₁^k to get v_s = Q_s(x₁^k)
   - Add IO cut at x₁^k

Stopping:
- Run 3–10 iterations; for a smoke test, 1–2 iterations is fine to validate plumbing.

---

## Implementation Notes (JuMP)
- Use a MILP solver (HiGHS works for MILP; Gurobi/CPLEX also fine).
- For LP relaxation step, either:
  - create a copy of the model with binaries relaxed, or
  - toggle variable categories temporarily if your code supports it.

---

## Minimal Success Criteria
- Your package can build the 2-stage tree with 2 scenarios.
- A forward pass returns a binary x₁.
- Backward pass:
  - can solve LP relaxation and extract π for z₂ = x₁
  - can solve the MIP L_s(π_s)
  - can add at least one cut at stage 1 without errors
