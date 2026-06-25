---
title: "CCMAlgorithm<T>"
description: "CCM — Convergent Cross-Mapping for detecting causation in nonlinear dynamical systems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.TimeSeries`

CCM — Convergent Cross-Mapping for detecting causation in nonlinear dynamical systems.

## For Beginners

CCM tests causation by checking whether one variable's history
can "predict" another variable using nearest-neighbor reconstruction in delay-coordinate
space. Crucially, if X causes Y, then Y's history cross-maps to X (not the other way),
which is the opposite of Granger causality's logic.

## How It Works

CCM is based on Takens' theorem from dynamical systems theory. If X causes Y, then
the shadow manifold reconstructed from Y should contain information about X, and
cross-mapping accuracy should improve (converge) with longer time series.

**Algorithm:**

- For each pair (i,j), construct delay embedding M_j from variable j with lag tau and dimension E
- For each point in M_j, find E+1 nearest neighbors in M_j
- Compute simplex weights from distances: w_k = exp(-d_k / d_1)
- Cross-map: predict x_i(t) as weighted combination of x_i at neighbor times
- Compute correlation rho between predicted and actual x_i
- If rho converges (improves) with increasing library size L, j cross-maps i → i causes j

Reference: Sugihara et al. (2012), "Detecting Causality in Complex Ecosystems", Science.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

