---
title: "BOSSAlgorithm<T>"
description: "BOSS (Best Order Score Search) — efficient permutation-based structure learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ScoreBased`

BOSS (Best Order Score Search) — efficient permutation-based structure learning.

## For Beginners

BOSS is a modern, fast algorithm that finds causal structures
by efficiently searching through possible variable orderings. For each variable, it
finds the single best position in the ordering, making large improvements per step.

## How It Works

BOSS combines permutation search with score-based evaluation. It maintains a variable
ordering and iteratively improves it by moving each variable to the position in the
ordering that maximizes the total BIC score. This "best position" operation is the
key difference from GRaSP's adjacent transpositions.

**Algorithm:**

- Initialize with a variable ordering based on marginal variance (ascending)
- For each variable v in the ordering:
- Remove v from its current position
- Try inserting v at every possible position (0 to d-1)
- Place v at the position that maximizes total BIC score
- Repeat the full pass until the ordering stabilizes
- Extract the DAG from the final ordering using greedy parent selection

Reference: Andrews et al. (2022), "Fast Scalable and Accurate Discovery of DAGs
Using the Best Order Score Search and Grow-Shrink Trees".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BOSSAlgorithm(CausalDiscoveryOptions)` | Initializes BOSS with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOrderingScore(Matrix<>,List<Int32>)` | Computes the total BIC score for a given ordering using greedy parent selection. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `ExtractDAG(Matrix<>,List<Int32>,Int32,Int32)` | Extracts the final DAG from the optimal ordering. |
| `SortByMarginalVariance(Matrix<>,List<Int32>,Int32)` | Sorts the ordering by marginal variance (ascending — exogenous variables first). |

