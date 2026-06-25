---
title: "DirectLiNGAMAlgorithm<T>"
description: "DirectLiNGAM — direct method for LiNGAM without ICA."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Functional`

DirectLiNGAM — direct method for LiNGAM without ICA.

## For Beginners

DirectLiNGAM finds causal structure step by step. First, it finds
the variable that seems to cause others but isn't caused by anything (the "root").
Then it removes that variable's influence and repeats. This gives a causal ordering
from which the full structure follows naturally.

## How It Works

DirectLiNGAM avoids the ICA step entirely and instead uses a direct regression-based
approach to identify the causal ordering. It iteratively finds the "root" variable
(the one with the most independent residuals) and removes its effect.

**Algorithm:**

- Find the variable with minimum dependence on others (root cause)
- Regress out the root's effect from all remaining variables
- Repeat on the residuals until all variables are ordered
- Estimate connection strengths via OLS in the causal order

Reference: Shimizu et al. (2011), "DirectLiNGAM: A Direct Method for Learning a
Linear Non-Gaussian Structural Equation Model", JMLR.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DirectLiNGAMAlgorithm(CausalDiscoveryOptions)` | Initializes DirectLiNGAM with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

