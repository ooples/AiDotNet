---
title: "NoCurlAlgorithm<T>"
description: "NoCurl — DAG learning via curl-free constraints on the graph structure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

NoCurl — DAG learning via curl-free constraints on the graph structure.

## For Beginners

NoCurl avoids the expensive matrix exponential used by NOTEARS.
Instead, it ensures acyclicity by finding a variable ordering (X before Y before Z)
and only allowing edges that follow this ordering. This is faster and simpler while
still finding good causal structures.

## How It Works

NoCurl parameterizes the DAG using a variable ordering and restricted edge weights.
Given an ordering pi, the adjacency matrix W is constrained to have edges only from
earlier to later variables in the ordering — guaranteeing acyclicity by construction.
The algorithm alternates between optimizing the ordering (via greedy swaps) and
optimizing edge weights (via OLS regression restricted to the ordering).

**Algorithm:**

- Initialize with an ordering based on marginal variance
- For the current ordering, compute optimal edge weights via OLS (restricted to the ordering)
- Apply L1 soft-thresholding for sparsity
- Try adjacent swaps to improve the L2 loss
- Accept the best swap and repeat
- Threshold small weights in the final W

Reference: Yu et al. (2021), "DAGs with No Curl: An Efficient DAG Structure Learning
Approach", ICML.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NoCurlAlgorithm(CausalDiscoveryOptions)` | Initializes NoCurl with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOrderingLoss(Matrix<>,Matrix<>,Int32[],Int32,Int32)` | Computes the total L2 loss for a given ordering. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `InitializeOrdering(Matrix<>,Int32)` | Initializes ordering based on diagonal of S (marginal variance, ascending). |

