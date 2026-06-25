---
title: "CAMAlgorithm<T>"
description: "CAM (Causal Additive Model) — order-based causal discovery with additive nonparametric regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Functional`

CAM (Causal Additive Model) — order-based causal discovery with additive nonparametric regression.

## For Beginners

CAM figures out the causal order of variables by finding which
variable is best predicted by the others using flexible (nonlinear) functions. It then
trims weak connections. Unlike linear methods, CAM can discover relationships where
the effect of one variable on another is curved or nonlinear.

## How It Works

CAM discovers causal structure in two stages:

- **Ordering:** Greedily selects the next variable that minimizes residual variance

when regressed on already-ordered variables using additive (kernel-smoothed) regression.

- **Pruning:** For each variable, tests whether each parent's additive contribution

is significant by comparing residual variance with and without that parent (likelihood ratio).

The model assumes X_j = Σ_k f_k(X_pa(j)_k) + ε_j where f_k are smooth nonparametric
functions estimated via Nadaraya–Watson kernel regression.

Reference: Buhlmann et al. (2014), "CAM: Causal Additive Models, High-Dimensional
Order Search and Penalized Regression", Annals of Statistics.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

