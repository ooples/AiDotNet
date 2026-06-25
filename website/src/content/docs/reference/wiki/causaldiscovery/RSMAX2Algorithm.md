---
title: "RSMAX2Algorithm<T>"
description: "RSMAX2 — Restricted Maximization, a hybrid constraint-based + score-based algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Hybrid`

RSMAX2 — Restricted Maximization, a hybrid constraint-based + score-based algorithm.

## For Beginners

RSMAX2 is a general framework for combining any "candidate finder"
with any "best structure finder." It first quickly identifies which variables MIGHT be
connected, then carefully selects the best connections from those candidates.

## How It Works

RSMAX2 (Restricted Structural Maximization, 2-phase) is a general framework for
hybrid causal discovery. Phase 1 uses conditional independence tests to learn each
variable's candidate parent/child set (restricting the search space). Phase 2 uses
greedy hill climbing with BIC scoring restricted to the candidate sets.

**Algorithm:**

- **Restrict phase:** For each variable, find candidate parents/children

using conditional independence tests (MMPC-like forward-backward selection)

- **Maximize phase:** Starting from an empty graph, greedily add edges

between variables and their candidates that improve BIC score

- Enforce acyclicity by rejecting edge additions that create cycles
- Apply backward deletion to remove edges that no longer improve score

Reference: Scutari (2010), "Learning Bayesian Networks with the bnlearn R Package", JSS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RSMAX2Algorithm(CausalDiscoveryOptions)` | Initializes RSMAX2 with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |
| `FindCandidateSets(Matrix<>,Int32)` | Phase 1: Find candidate parent/child sets using conditional independence tests. |

