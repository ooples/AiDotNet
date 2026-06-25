---
title: "OCSEAlgorithm<T>"
description: "oCSE — optimal Causation Entropy for causal network inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.InformationTheoretic`

oCSE — optimal Causation Entropy for causal network inference.

## For Beginners

oCSE measures how much a variable helps predict another variable's
CHANGES over time (not just its values). This is closer to true causation — a cause
should affect how the effect changes.

## How It Works

oCSE uses causation entropy — a measure of the information a variable provides about
another variable's transition — to identify causal links. It greedily selects the
optimal conditioning set that maximizes the causation entropy criterion.

**Algorithm:**

- For each target Y, compute transition: delta_Y[t] = Y[t+1] - Y[t]
- For each candidate cause X, compute causation entropy:

CE(X→Y|S) = MI(delta_Y ; X | S) where S is the current conditioning set

- Greedily add variables to S that maximize CE until no variable exceeds threshold
- Variables in S are the discovered causal parents of Y
- Edge weight = causation entropy value (Gaussian MI approximation)

Reference: Sun et al. (2015), "Causal Network Inference by Optimal Causation Entropy",
SIAM Journal on Applied Dynamical Systems.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |
| `SupportsTimeSeries` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

