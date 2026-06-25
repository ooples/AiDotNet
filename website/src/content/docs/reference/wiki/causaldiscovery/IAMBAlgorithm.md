---
title: "IAMBAlgorithm<T>"
description: "IAMB (Incremental Association Markov Blanket) — efficient Markov blanket discovery."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ConstraintBased`

IAMB (Incremental Association Markov Blanket) — efficient Markov blanket discovery.

## For Beginners

IAMB finds the "neighborhood" (Markov blanket) of each variable —
the set of variables that directly influence it or are directly influenced by it.
It does this by adding the most relevant variables one at a time, then pruning any
that turn out to be redundant. The blankets are then combined into a causal graph.

## How It Works

IAMB discovers the causal structure by first finding the Markov blanket of each variable
using a two-phase approach:

- **Forward phase:** Greedily add the variable most associated with the target

(given the current blanket) until no more significantly associated variables exist

- **Backward phase:** Remove any variable from the blanket that becomes

conditionally independent of the target given the remaining blanket members

- Build the skeleton from pairwise Markov blanket membership
- Orient edges using v-structure detection and Meek rules

Reference: Tsamardinos et al. (2003), "Algorithms for Large Scale Markov Blanket Discovery".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IAMBAlgorithm(CausalDiscoveryOptions)` | Initializes IAMB with optional configuration. |

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
| `FindMarkovBlanket(Matrix<>,Int32,Int32)` | Finds the Markov blanket of a target variable using IAMB's forward-backward approach. |

