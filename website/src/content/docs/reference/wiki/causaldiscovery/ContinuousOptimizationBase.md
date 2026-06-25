---
title: "ContinuousOptimizationBase<T>"
description: "Base class for continuous optimization causal discovery methods (NOTEARS, DAGMA, GOLEM)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

Base class for continuous optimization causal discovery methods (NOTEARS, DAGMA, GOLEM).

## For Beginners

Traditional methods check all possible graph structures to find the best one,
which is extremely slow for many variables. Continuous optimization methods instead use calculus-based
optimization (like gradient descent) to smoothly search for the best graph, which is much faster.

## How It Works

These methods formulate the DAG learning problem as a continuous optimization:
minimize a loss function (e.g., least squares) subject to a smooth acyclicity constraint.
The key innovation is replacing the combinatorial DAG constraint with a differentiable function.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `HTolerance` | Convergence tolerance for the acyclicity constraint h(W). |
| `Lambda1` | L1 sparsity penalty (lambda1). |
| `LossType` | Loss type: "l2" (least squares), "logistic", or "poisson". |
| `MaxIterations` | Maximum number of outer loop iterations. |
| `WThreshold` | Edge weight threshold for post-optimization pruning. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyOptions(CausalDiscoveryOptions)` | Applies common options from `CausalDiscoveryOptions` to the algorithm parameters. |
| `ComputeL1Norm(Matrix<>)` | Computes the L1 norm of a matrix: sum of absolute values. |
| `ComputeL2Loss(Matrix<>,Matrix<>)` | Computes the L2 loss: (1/2n) * \|\|X - XW\|\|²_F and its gradient. |
| `ComputeNOTEARSConstraint(Matrix<>,Int32)` | Computes the NOTEARS acyclicity constraint h(W) = tr(e^(W∘W)) - d and its gradient dh/dW = 2 * e^(W∘W) ∘ W. |
| `FallbackCorrelationGraph(Matrix<>)` | Fallback: uses pairwise correlation to detect edges when continuous optimization fails to find structure (e.g., due to near-degenerate data). |
| `InvertWithRidge(Matrix<>,Int32,)` | Computes the precision matrix (inverse covariance) with ridge regularization using the library's Matrix type and numeric operations. |
| `StandardizeData(Matrix<>)` | Standardizes data to zero mean and unit variance per column, with a small column-specific perturbation to break exact collinearity. |
| `ThresholdAndClean(Matrix<>,Double)` | Applies threshold to W: sets entries with \|W[i,j]\| < threshold to 0. |
| `ThresholdWithFallback(Matrix<>,Double,Matrix<>)` | Thresholds the weight matrix, with covariance-based fallback when no edges survive. |

## Fields

| Field | Summary |
|:-----|:--------|
| `PerturbationCycleModulus` | Modulus used to cycle column-specific perturbations during data standardization. |

