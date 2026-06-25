---
title: "ANMAlgorithm<T>"
description: "ANM (Additive Noise Model) — pairwise causal discovery via independence of residuals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Functional`

ANM (Additive Noise Model) — pairwise causal discovery via independence of residuals.

## For Beginners

If X truly causes Y, then the "noise" left over after predicting
Y from X should have nothing to do with X. But if you try predicting X from Y, the
leftover noise will still be related to Y. ANM uses this asymmetry to figure out
which variable causes which.

## How It Works

ANM determines causal direction between variable pairs by fitting Y = f(X) + N
in both directions and checking which direction yields residuals N that are independent
of the cause. If the residuals from X → Y are more independent of X than the residuals
from Y → X are of Y, then X → Y is the inferred causal direction.

Reference: Hoyer et al. (2008), "Nonlinear Causal Discovery with Additive Noise Models",
NIPS.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

