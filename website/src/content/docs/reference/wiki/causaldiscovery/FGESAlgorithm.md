---
title: "FGESAlgorithm<T>"
description: "FGES (Fast Greedy Equivalence Search) — greedy DAG search with BIC score caching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ScoreBased`

FGES (Fast Greedy Equivalence Search) — greedy DAG search with BIC score caching.

## For Beginners

FGES searches for the best graph by first greedily adding edges
that improve the score, then removing edges that improve the score. Caching remembered
scores avoids recalculating the same parent-set configurations.

## How It Works

This implementation performs greedy forward (edge addition) and backward (edge removal)
search over DAG structures with BIC score caching for efficiency. It uses the same
forward-backward structure as GES but with cached score evaluations to avoid redundant
BIC computations.

Reference: Ramsey et al. (2017), "A Million Variables and More", International Journal of Data Science and Analytics.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

