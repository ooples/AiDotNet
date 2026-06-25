---
title: "RFCIAlgorithm<T>"
description: "RFCI (Really Fast Causal Inference) — scalable FCI for large datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ConstraintBased`

RFCI (Really Fast Causal Inference) — scalable FCI for large datasets.

## For Beginners

RFCI does the same thing as FCI (finds causal relationships
even with hidden variables) but much faster. It achieves this by being smarter about
which statistical tests to run, skipping tests that are unlikely to change the result.

## How It Works

RFCI speeds up FCI by reducing the number of conditional independence tests.
Instead of testing all possible conditioning sets (as FCI does), RFCI:

- Runs a PC-like skeleton phase with limited conditioning set sizes
- For possible v-structures, only tests conditioning sets from adjacencies

(not all possible subsets), reducing complexity from exponential to polynomial

- Uses discriminating path rules more sparingly
- Produces a PAG (Partial Ancestral Graph) that accounts for latent confounders

Reference: Colombo et al. (2012), "Learning High-Dimensional DAGs with Latent
and Selection Variables", AOAS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RFCIAlgorithm(CausalDiscoveryOptions)` | Initializes RFCI with optional configuration. |

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
| `GetAdjacencies(Boolean[0:,0:],Int32,Int32,Int32)` | Gets adjacencies of node i, excluding node exclude. |

