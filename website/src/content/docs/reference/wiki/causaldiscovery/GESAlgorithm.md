---
title: "GESAlgorithm<T>"
description: "GES (Greedy Equivalence Search) — score-based causal discovery over equivalence classes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ScoreBased`

GES (Greedy Equivalence Search) — score-based causal discovery over equivalence classes.

## For Beginners

GES builds a causal graph by first adding edges that improve the
model fit, then removing edges that are unnecessary. It uses a score (BIC) that rewards
fitting the data well while penalizing too many edges.

## How It Works

GES searches over Markov equivalence classes of DAGs using two phases:

- **Forward phase:** Greedily adds edges that most improve the BIC score.
- **Backward phase:** Greedily removes edges that improve the BIC score.

Reference: Chickering (2002), "Optimal Structure Identification with Greedy Search", JMLR.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

