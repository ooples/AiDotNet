---
title: "GFCIAlgorithm<T>"
description: "GFCI — Greedy FCI, a hybrid of GES and FCI."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Hybrid`

GFCI — Greedy FCI, a hybrid of GES and FCI.

## For Beginners

GFCI is useful when you suspect there are hidden variables affecting
your data. It first quickly finds a good graph structure, then checks whether some
connections might actually be due to hidden common causes rather than direct effects.

## How It Works

GFCI combines score-based and constraint-based approaches to handle latent confounders.
Phase 1 uses greedy score-based search (like GES) to find an initial skeleton and
orientations. Phase 2 applies FCI-like rules to detect possible latent confounders
and convert directed edges to bidirected edges where appropriate.

**Algorithm:**

- **Score-based phase:** Use greedy hill climbing with BIC to find an initial DAG
- **Skeleton validation:** Test remaining edges with CI tests; remove if independent
- **Latent detection:** For each unshielded triple i — k — j where i and j are

non-adjacent, check if k is in the separation set. If not, orient as collider (i → k ← j)

- **Discriminating path check:** Identify possible bidirected edges (latent confounders)

by checking if edges oriented in both directions can be explained by a common cause

- Apply FCI orientation rules to propagate orientations

Reference: Ogarrio et al. (2016), "A Hybrid Causal Search Algorithm for Latent
Variable Models", PGM.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GFCIAlgorithm(CausalDiscoveryOptions)` | Initializes GFCI with optional configuration. |

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

