---
title: "GRaSPAlgorithm<T>"
description: "GRaSP (Greedy Relaxation of Sparsest Permutation) — permutation-based causal discovery."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ScoreBased`

GRaSP (Greedy Relaxation of Sparsest Permutation) — permutation-based causal discovery.

## For Beginners

GRaSP tries different orderings of variables and for each ordering,
finds the simplest (sparsest) causal graph. It's designed to find graphs with fewer
edges, which often corresponds to the true causal structure.

## How It Works

GRaSP searches over permutations (orderings) of variables and selects the sparsest
DAG consistent with each ordering. It uses greedy local moves (adjacent transpositions)
to explore the permutation space, accepting moves that reduce the total number of edges.

**Algorithm:**

- Initialize with an ordering (e.g., based on marginal variance)
- For the current ordering, compute the optimal parent set for each variable

(parents must precede the variable in the ordering)

- Count total edges (sparsity measure)
- Try all adjacent transpositions (swap positions i and i+1)
- For each swap, recompute parent sets for the two affected variables
- Accept the swap that yields the largest reduction in total edges (or best BIC if tied)
- Repeat until no swap improves sparsity

Reference: Lam et al. (2022), "Greedy Relaxations of the Sparsest Permutation Algorithm".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GRaSPAlgorithm(CausalDiscoveryOptions)` | Initializes GRaSP with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOptimalParents(Matrix<>,Int32[])` | For each variable in the ordering, finds the best parent set from predecessors using BIC. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `InitializeOrdering(Matrix<>,Int32)` | Initializes ordering based on marginal variance (ascending — low variance first). |

