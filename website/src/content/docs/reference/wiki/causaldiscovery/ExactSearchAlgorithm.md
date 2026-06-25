---
title: "ExactSearchAlgorithm<T>"
description: "Exact Search (Dynamic Programming) — optimal DAG structure learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ScoreBased`

Exact Search (Dynamic Programming) — optimal DAG structure learning.

## For Beginners

This algorithm finds the absolute best causal graph by
systematically checking all possibilities using clever math shortcuts (dynamic
programming). It's guaranteed to find the optimal solution but only works for
small datasets (up to about 20 variables due to O(2^d) complexity).

## How It Works

Uses dynamic programming over subsets of variables to find the globally optimal
DAG structure according to a decomposable score (BIC). The algorithm works in two phases:

**Algorithm (Silander-Myllymaki):**

- Phase 1 (Parent scoring): For each variable and each subset of potential parents,

compute the BIC score. Store the best parent set for each variable given each ancestor set.

- Phase 2 (Order search): Use DP over variable orderings.

For each subset S, compute the best DAG score by trying each variable as the "last" in
the ordering. The best parents for that variable come from S minus that variable.

- Backtrack through the DP table to reconstruct the optimal parent assignments.

Reference: Silander and Myllymaki (2006), "A Simple Approach for Finding the
Globally Optimal Bayesian Network Structure".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExactSearchAlgorithm(CausalDiscoveryOptions)` | Initializes Exact Search with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BitCount(Int32)` | Counts the number of set bits in an integer (population count). |
| `BitmaskToSet(Int32)` | Converts a bitmask to a HashSet of variable indices. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `EstimateMultivariateOLS(Matrix<>,Int32,List<Int32>,Int32)` | Multivariate OLS: regress child on all parents jointly. |

