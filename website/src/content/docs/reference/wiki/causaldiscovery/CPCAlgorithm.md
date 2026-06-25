---
title: "CPCAlgorithm<T>"
description: "CPC (Conservative PC) — PC variant that avoids erroneous v-structure orientation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ConstraintBased`

CPC (Conservative PC) — PC variant that avoids erroneous v-structure orientation.

## For Beginners

Standard PC sometimes incorrectly orients edges when the
independence tests are noisy. CPC is more careful — it only orients an edge
when ALL evidence agrees on the direction. This means fewer edges are oriented,
but the orientations that remain are more reliable.

## How It Works

CPC modifies the PC algorithm's orientation phase to be more conservative. Before
orienting a triple i — k — j as a v-structure (i → k ← j), CPC checks ALL possible
subsets of the adjacency of i (and j) that could serve as separation sets. A v-structure
is only oriented if k is NEVER in any separation set (definite non-collider) or ALWAYS
in every separation set (definite collider). Ambiguous triples are left unoriented.

**Algorithm:**

- Run PC skeleton phase (same as standard PC)
- For each unshielded triple i — k — j:
- Collect ALL subsets of adj(i)\{j} and adj(j)\{i} up to MaxConditioningSetSize
- Test CI(i, j | S) for each subset S
- Classify k as: definite collider (never in any separating set),

definite non-collider (always in every separating set), or ambiguous

- Only orient v-structure if k is a definite collider
- Apply Meek orientation rules

Reference: Ramsey et al. (2012), "Adjacency-Faithfulness and Conservative
Causal Inference", UAI.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CPCAlgorithm(CausalDiscoveryOptions)` | Initializes CPC with optional configuration. |

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
| `IsDefiniteCollider(Matrix<>,Int32,Int32,Int32,Boolean[0:,0:],Int32)` | Checks if k is a definite collider in the triple i — k — j per CPC rules. |

