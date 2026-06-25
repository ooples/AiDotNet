---
title: "PartitionMCMCAlgorithm<T>"
description: "Partition MCMC — MCMC sampling over DAG partitions for structure learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Bayesian`

Partition MCMC — MCMC sampling over DAG partitions for structure learning.

## For Beginners

This method groups variables into "layers" (partitions) where
variables in earlier layers can cause variables in later layers but not vice versa.
It explores different layer arrangements to find plausible causal structures.

## How It Works

Partition MCMC extends Order MCMC by sampling over partitions of variables instead
of total orderings. A partition groups variables into layers; edges can go from earlier
layers to later layers. The partition space is between orderings and DAGs in granularity,
providing better mixing than Order MCMC while remaining efficient.

**Algorithm:**

- Initialize: each variable in its own partition element (= total ordering)
- Propose moves: split a partition element, merge two adjacent elements, or swap elements
- For each partition, compute the optimal DAG via BIC-greedy parent selection
- Accept/reject via Metropolis-Hastings with BIC-based scoring
- After burn-in, accumulate edge posterior probabilities
- Return edges with posterior probability > 0.5

Reference: Kuipers and Moffa (2017), "Partition MCMC for Inference on Acyclic
Digraphs", JASA.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

