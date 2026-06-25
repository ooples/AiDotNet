---
title: "MMPCAlgorithm<T>"
description: "MMPC (Max-Min Parents and Children) — identifies the parents and children of each variable."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ConstraintBased`

MMPC (Max-Min Parents and Children) — identifies the parents and children of each variable.

## For Beginners

MMPC finds the "direct neighbors" of each variable in the
causal graph. It's faster than PC for large graphs because it works locally
(one variable at a time) rather than globally.

## How It Works

MMPC identifies the parents and children (direct causes and effects) of each variable
using a forward-backward selection procedure based on conditional independence tests.

Reference: Tsamardinos et al. (2003), "Algorithms for Large Scale Markov Blanket Discovery".

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsLatentConfounders` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

