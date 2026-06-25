---
title: "GOBNILPAlgorithm<T>"
description: "GOBNILP — Globally Optimal Bayesian Network learning using Integer Linear Programming."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Specialized`

GOBNILP — Globally Optimal Bayesian Network learning using Integer Linear Programming.

## For Beginners

GOBNILP guarantees finding the BEST possible graph according to the
scoring criterion. Most other algorithms are heuristic (they find good but not necessarily
optimal solutions). The trade-off is that GOBNILP can be slow for many variables.

## How It Works

GOBNILP formulates Bayesian network structure learning as an integer linear programming (ILP)
problem. It finds the globally optimal DAG by:
(1) Pre-computing BIC scores for all candidate parent sets per variable,
(2) Formulating a 0-1 ILP where binary variables indicate which parent set is selected
for each variable,
(3) Enforcing acyclicity via cluster constraints (for each subset S, at least one variable
in S must have all its parents outside S),
(4) Solving the ILP with a branch-and-bound search with lazy constraint generation.

**Algorithm:**

- For each variable j and each candidate parent set P ⊆ V\{j} with |P| ≤ maxParents,

compute BIC score: score(j, P)

- Create binary variable y_{j,P} = 1 iff P is the parent set of j
- Objective: maximize sum_j sum_P score(j, P) * y_{j,P}
- Constraint: for each j, exactly one parent set is selected: sum_P y_{j,P} = 1
- Acyclicity: for each cluster C ⊆ V with |C| ≥ 2, at least one j ∈ C has parent set

P with P ∩ C = ∅ (enforced lazily via cycle detection)

- Solve via branch-and-bound with lazy acyclicity constraints

Reference: Cussens (2012), "Bayesian Network Learning with Cutting Planes", UAI.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

