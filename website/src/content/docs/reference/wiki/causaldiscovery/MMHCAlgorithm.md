---
title: "MMHCAlgorithm<T>"
description: "MMHC — Max-Min Hill-Climbing, a hybrid constraint-based + score-based algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Hybrid`

MMHC — Max-Min Hill-Climbing, a hybrid constraint-based + score-based algorithm.

## For Beginners

MMHC first quickly identifies which variables MIGHT be related
(using statistical tests), then carefully determines the exact direction and strength
of those relationships (using a scoring approach). This makes it both fast and accurate.

## How It Works

MMHC combines two phases:

- **MMPC phase (constraint-based):** For each variable, identify candidate parents/children

using the Max-Min Parents and Children heuristic. This restricts the search space.

- **HC phase (score-based):** Run greedy Hill Climbing search within the restricted

space defined by the MMPC skeleton, optimizing BIC score.

Reference: Tsamardinos et al. (2006), "The Max-Min Hill-Climbing Bayesian Network
Structure Learning Algorithm", Machine Learning.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMinAssociation(Matrix<>,Int32,Int32,List<Int32>)` | Computes minimum association (absolute correlation) over conditioning subsets. |
| `ComputePartialCorrelationSingle(Matrix<>,Int32,Int32,Int32)` | Computes partial correlation conditioned on a single variable. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `HillClimbPhase(Matrix<>,Boolean[0:,0:])` | Hill Climbing phase within the skeleton-restricted space. |
| `MMPCPhase(Matrix<>)` | MMPC phase: for each variable, identify candidate neighbors using Max-Min heuristic. |
| `WouldCreateCycle(List<Int32>[],Int32,Int32,Int32)` | Checks if adding edge from → to would create a cycle. |

