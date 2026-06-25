---
title: "LinearMixedModelOptions<T>"
description: "Configuration options for mixed-effects (hierarchical/multilevel) models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for mixed-effects (hierarchical/multilevel) models.

## For Beginners

Mixed-effects models are for data that has "groups" or "clusters".

Examples of grouped data:

- Students nested in schools
- Patients nested in hospitals
- Repeated measurements on individuals
- Products sold in different stores

Why use mixed models instead of regular regression?

1. Properly accounts for non-independence within groups
2. Gives you valid standard errors and p-values
3. Shares information across groups (partial pooling)
4. Estimates how much variation is between vs. within groups

## How It Works

Mixed-effects models handle data with natural grouping or clustering by estimating
both population-level (fixed) effects and group-level (random) effects. They properly
account for correlation within groups and provide valid inference.

## Properties

| Property | Summary |
|:-----|:--------|
| `BoundVarianceComponents` | Gets or sets whether to use the bounded optimization for variance components. |
| `ComputeBLUPs` | Gets or sets whether to compute predicted random effects (BLUPs). |
| `ComputeRSquared` | Gets or sets whether to compute marginal and conditional R-squared. |
| `ComputeVarianceCI` | Gets or sets whether to compute confidence intervals for variance components. |
| `EstimationMethod` | Gets or sets the estimation method. |
| `Optimizer` | Gets or sets the optimizer to use. |
| `UseSVD` | Gets or sets whether to use singular value decomposition for numerical stability. |
| `Verbose` | Gets or sets whether to print verbose output during fitting. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_confidenceLevel` | Gets or sets the confidence level for intervals. |
| `_maxIterations` | Gets or sets the maximum number of iterations for the optimization. |
| `_tolerance` | Gets or sets the convergence tolerance. |

