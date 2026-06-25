---
title: "PCAlgorithm<T>"
description: "PC Algorithm — constraint-based causal discovery using conditional independence tests."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ConstraintBased`

PC Algorithm — constraint-based causal discovery using conditional independence tests.

## For Beginners

PC figures out which variables cause which by testing:
"Are X and Y still related after we account for other variables?" If X and Y
become unrelated (conditionally independent) given some set of variables, the
edge between them is removed. Then remaining edges are oriented to form a DAG.

## How It Works

The PC (Peter-Clark) algorithm learns a causal graph by:

- Starting with a complete undirected graph
- Removing edges between conditionally independent variable pairs
- Orienting edges using v-structures and orientation rules

Reference: Spirtes, Glymour, and Scheines (2000), "Causation, Prediction, and Search".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PCAlgorithm(CausalDiscoveryOptions)` | Initializes the PC algorithm with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsLatentConfounders` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyMeekRules(Boolean[0:,0:],Boolean[0:,0:],Int32)` | Applies Meek orientation rules R1–R3 iteratively until no more edges can be oriented. |
| `DiscoverStructureCore(Matrix<>)` |  |

