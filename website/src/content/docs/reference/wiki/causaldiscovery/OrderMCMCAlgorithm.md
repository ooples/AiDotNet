---
title: "OrderMCMCAlgorithm<T>"
description: "Order MCMC — MCMC sampling over variable orderings for Bayesian structure learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Bayesian`

Order MCMC — MCMC sampling over variable orderings for Bayesian structure learning.

## For Beginners

Instead of searching over all possible graphs (very hard), this
method searches over all possible orderings of variables (much easier). Once you know
the order, finding the best graph is straightforward.

## How It Works

Order MCMC samples from the posterior over variable orderings (rather than over DAGs
directly). Given an ordering, the optimal DAG can be computed efficiently via greedy
parent selection with BIC scoring. The Markov chain proposes moves by swapping adjacent
elements in the ordering, accepting with Metropolis-Hastings probability.

**Algorithm:**

- Initialize a random variable ordering
- Propose a new ordering by swapping two adjacent elements
- Compute optimal DAGs for both orderings using greedy BIC parent selection
- Accept/reject via Metropolis-Hastings: accept with prob min(1, exp(score_new - score_old))
- After burn-in, accumulate edge frequencies across accepted samples
- Return edges that appear in more than 50% of posterior samples

Reference: Friedman and Koller (2003), "Being Bayesian About Network Structure", MLJ.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMultivariateOLSWeights(Matrix<>,Int32[],Int32)` | Computes multivariate OLS regression weights: beta = Cov(parents,parents)^{-1} * Cov(parents,target). |
| `ComputeOrderingScore(Matrix<>,Int32[],Int32)` | Computes total negative BIC for an ordering (higher is better). |
| `DiscoverStructureCore(Matrix<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `MinEdgeWeight` | Minimum absolute weight for an edge to be included in the DAG. |
| `PivotTolerance` | Pivot tolerance for detecting singular/near-singular matrices during Gaussian elimination. |
| `RegularizationEpsilon` | Regularization epsilon for numerical stability in matrix inversion. |

