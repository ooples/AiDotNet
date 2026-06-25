---
title: "GOLEMAlgorithm<T>"
description: "GOLEM — Gradient-based Optimization with Likelihood for structure learning of linear DAGs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

GOLEM — Gradient-based Optimization with Likelihood for structure learning of linear DAGs.

## For Beginners

GOLEM is an alternative to NOTEARS that's simpler to implement and
can be more efficient. Instead of using a special "acyclicity constraint," it bakes the
constraint into the objective function itself. It directly measures how likely the data is
given a particular causal graph, plus a penalty for cycles.

## How It Works

GOLEM uses a likelihood-based score function with a soft DAG penalty. Unlike NOTEARS which
uses a hard acyclicity constraint via augmented Lagrangian, GOLEM optimizes a penalized
likelihood objective directly, avoiding the expensive inner-outer loop structure.

**Two variants:**

- **GOLEM-EV (Equal Variance):** Assumes equal noise variance across all variables.

Score = n*d/2 * log(||X - XW||²_F) - log|det(I - W)| + lambda1*||W||_1

- **GOLEM-NV (Non-equal Variance):** Allows different noise variances.

Score = sum_j [n/2 * log(||X_j - XW_j||²) - log|det(I - W)|/d] + lambda1*||W||_1

**Key advantage:** Single-loop optimization (no inner/outer loops), uses standard
gradient descent optimizers (Adam), and the log-determinant term naturally penalizes cycles.

Reference: Ng et al. (2020), "On the Role of Sparsity and DAG Constraints for Learning
Linear DAGs", NeurIPS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GOLEMAlgorithm(CausalDiscoveryOptions,Boolean)` | Initializes GOLEM with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDAGPenalty(Matrix<>,Int32)` | Computes a soft DAG penalty: h(W) = tr(e^(W∘W)) - d. |
| `ComputeGOLEMEV(Matrix<>,Matrix<>,Int32,Int32)` | GOLEM-EV score: equal variance assumption. |
| `ComputeGOLEMNV(Matrix<>,Matrix<>,Int32,Int32)` | GOLEM-NV score: non-equal variance assumption. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `GetLastRunInfo` | Gets the iteration count and convergence info from the last run. |

