---
title: "TabuSearchAlgorithm<T>"
description: "Tabu Search — score-based DAG learning with memory to escape local optima."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ScoreBased`

Tabu Search — score-based DAG learning with memory to escape local optima.

## For Beginners

Tabu search is like hill climbing but with a memory. If you just
climbed from point A to point B, you're not allowed to go back to A for a while.

## How It Works

Tabu search extends hill climbing by maintaining a "tabu list" of recently visited
states that cannot be revisited.

Reference: Glover (1989, 1990), "Tabu Search".

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

