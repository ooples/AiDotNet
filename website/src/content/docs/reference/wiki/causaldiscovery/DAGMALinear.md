---
title: "DAGMALinear<T>"
description: "DAGMA Linear — DAG learning via M-matrices and a log-determinant acyclicity characterization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

DAGMA Linear — DAG learning via M-matrices and a log-determinant acyclicity characterization.

## For Beginners

DAGMA does the same thing as NOTEARS (finding causal relationships)
but uses a different math trick that's faster and more numerically stable. Instead of
using the matrix exponential, it uses the log-determinant, which has nicer gradients
and converges faster — especially for larger problems.

## How It Works

DAGMA replaces the NOTEARS matrix exponential constraint with a log-determinant constraint:
h(W, s) = -log det(sI - W∘W) + d*log(s), which has better gradient behavior and is ~10x faster.

**Optimization:** Uses a central path / barrier method instead of augmented Lagrangian.
At each outer iteration, the domain parameter s decreases, tightening the DAG constraint.
Inner optimization uses gradient descent with Adam optimizer.

**Default hyperparameters (from original paper):**
lambda1 = 0.03, T = 5, mu_init = 1.0, mu_factor = 0.1,
s = [1.0, 0.9, 0.8, 0.7, 0.6], lr = 0.0003, w_threshold = 0.3

Reference: Bello et al. (2022), "DAGMA: Learning DAGs via M-matrices and a Log-Determinant
Acyclicity Characterization", NeurIPS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DAGMALinear(CausalDiscoveryOptions)` | Initializes DAGMA Linear with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLogDetConstraint(Matrix<>,Double,Int32)` | Computes the DAGMA log-det acyclicity constraint and gradient. |
| `ComputeLogDeterminant(Matrix<>,Int32)` | Computes log-determinant using LU decomposition. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `GetLastRunInfo` | Gets the iteration count and convergence info from the last run. |
| `HasNegativeEntry(Matrix<>,Int32)` | Checks if any entry of the matrix is negative (M-matrix violation). |
| `InvertMatrix(Matrix<>,Int32)` | Inverts a matrix using Gauss-Jordan elimination. |
| `SolveInnerProblem(Matrix<>,Matrix<>,Double,Double,Int32,Int32)` | Solves the inner optimization problem using Adam optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `ADAM_BETA1` | Adam optimizer beta1 (first moment decay). |
| `ADAM_BETA2` | Adam optimizer beta2 (second moment decay). |
| `CHECKPOINT_INTERVAL` | Checkpoint interval for convergence checking. |
| `DAGMA_DEFAULT_LAMBDA1` | Default L1 penalty for DAGMA (different from NOTEARS default). |
| `DEFAULT_LEARNING_RATE` | Learning rate for the Adam optimizer in the inner loop. |
| `DEFAULT_MAX_ITER` | Number of inner iterations for the final outer step. |
| `DEFAULT_MU_FACTOR` | Multiplicative decay factor for mu between outer iterations. |
| `DEFAULT_MU_INIT` | Initial penalty weight for the log-det constraint. |
| `DEFAULT_T` | Number of outer iterations (central path steps). |
| `DEFAULT_WARM_ITER` | Number of warm-up inner iterations for non-final outer steps. |
| `INNER_CONVERGENCE_TOL` | Convergence tolerance for the inner loop objective change. |

