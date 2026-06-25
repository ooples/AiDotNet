---
title: "NOTEARSLinear<T>"
description: "NOTEARS Linear — continuous optimization for DAG structure learning with linear relationships."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

NOTEARS Linear — continuous optimization for DAG structure learning with linear relationships.

## For Beginners

NOTEARS finds which variables cause which other variables by solving
a math optimization problem. It learns a weight matrix W where W[i,j] != 0 means "variable i
directly causes variable j." The clever trick is a special math formula (the trace of a matrix
exponential) that equals zero if and only if the graph has no cycles — which is required for
a valid causal graph. This avoids checking every possible graph structure.

## How It Works

NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for
Structure learning) reformulates the combinatorial DAG constraint as a smooth equality constraint:

**Objective:** minimize F(W) = (1/2n)||X - XW||²_F + lambda1 * ||W||_1

**Subject to:** h(W) = tr(e^(W∘W)) - d = 0 (acyclicity constraint)

The acyclicity constraint h(W) is zero if and only if W encodes a DAG. The optimization
uses an augmented Lagrangian method with L-BFGS inner solver.

**Algorithm (matching original paper):**

- Initialize W = 0, alpha = 0, rho = 1
- Inner loop: minimize augmented Lagrangian using L-BFGS with bounds [0, inf)
- If h(W_new) > 0.25 * h(W_old): rho *= 10, repeat inner loop
- Else: update alpha += rho * h, proceed
- Stop when h < h_tol or rho >= rho_max
- Threshold small weights: W[|W| < w_threshold] = 0

**Default hyperparameters (from original paper):**
lambda1 = 0.1, max_iter = 100, h_tol = 1e-8, rho_max = 1e+16, w_threshold = 0.3

Reference: Zheng et al. (2018), "DAGs with NO TEARS: Continuous Optimization for Structure Learning",
Advances in Neural Information Processing Systems (NeurIPS).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NOTEARSLinear(CausalDiscoveryOptions)` | Initializes a new instance of NOTEARSLinear with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAugmentedLagrangianGradient(Matrix<>,Matrix<>,Double,Double,Int32)` | Computes the gradient of the augmented Lagrangian (without L1 term, which is handled separately). |
| `ComputeAugmentedLagrangianObjective(Matrix<>,Matrix<>,Double,Double)` | Computes the augmented Lagrangian objective value (without L1). |
| `ComputeLBFGSDirection(Double[],List<Double[]>,List<Double[]>)` | L-BFGS two-loop recursion to compute search direction. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `GetLastRunInfo` | Gets the iteration count and convergence info from the last run. |
| `SolveLBFGSSubproblem(Matrix<>,Matrix<>,Double,Double,Int32)` | Solves the augmented Lagrangian subproblem using L-BFGS. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CONSTRAINT_REDUCTION_THRESHOLD` | Threshold for determining if the constraint has been sufficiently reduced. |
| `DEFAULT_RHO_INIT` | Initial penalty parameter for the augmented Lagrangian. |
| `DEFAULT_RHO_MAX` | Maximum penalty parameter before termination. |
| `INNER_MAX_ITERATIONS` | Maximum number of L-BFGS iterations for the inner optimization loop. |
| `INNER_TOLERANCE` | Convergence tolerance for L-BFGS inner loop. |
| `LBFGS_LEARNING_RATE` | Learning rate for gradient descent steps within L-BFGS. |
| `LBFGS_MEMORY` | L-BFGS memory size (number of past iterations to store). |
| `RHO_MULTIPLY_FACTOR` | Factor by which rho is increased when the constraint is not sufficiently reduced. |

