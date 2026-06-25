---
title: "ScoreBasedBase<T>"
description: "Base class for score-based causal discovery algorithms (GES, FGES, Hill Climbing, Tabu, etc.)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalDiscovery.ScoreBased`

Base class for score-based causal discovery algorithms (GES, FGES, Hill Climbing, Tabu, etc.).

## For Beginners

Score-based methods give each possible causal graph a "grade"
(score) based on how well it explains the data. They then search for the graph
with the best grade. Higher scores mean the graph fits the data better while
remaining simple (not too many edges).

## How It Works

Score-based methods search over the space of DAGs by evaluating each candidate graph
using a scoring criterion (typically BIC or BDeu). They use search strategies like
greedy equivalence search, hill climbing, or tabu search to find high-scoring graphs.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `MaxIterations` | Maximum number of search iterations. |
| `MaxParents` | Maximum number of parents per node. |
| `PenaltyDiscount` | BIC penalty discount factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyScoreOptions(CausalDiscoveryOptions)` | Applies options from CausalDiscoveryOptions. |
| `ComputeAbsCorrelation(Matrix<>,Int32,Int32)` | Computes the absolute Pearson correlation between two columns of data. |
| `ComputeBIC(Matrix<>,Int32,HashSet<Int32>)` | Computes the BIC score for a variable given its parents. |
| `WouldCreateCycle(HashSet<Int32>[],Int32,Int32)` | Checks if adding an edge from parent to child would create a cycle. |

