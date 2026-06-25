---
title: "HillClimbingAlgorithm<T>"
description: "Hill Climbing — greedy score-based DAG structure learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ScoreBased`

Hill Climbing — greedy score-based DAG structure learning.

## For Beginners

Hill climbing is the simplest score-based approach. At each step,
it tries adding, removing, or flipping every possible edge and picks the change that
improves the score the most.

## How It Works

Hill climbing searches over individual DAGs by greedily applying the single-edge operation
(add, remove, or reverse) that most improves the BIC score.

Reference: Heckerman et al. (1995), "Learning Bayesian Networks".

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

