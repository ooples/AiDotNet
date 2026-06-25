---
title: "PCNOTEARSAlgorithm<T>"
description: "PC-NOTEARS — Hybrid of PC skeleton discovery with NOTEARS continuous optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Hybrid`

PC-NOTEARS — Hybrid of PC skeleton discovery with NOTEARS continuous optimization.

## For Beginners

This hybrid first uses statistical tests to quickly figure out which
variable pairs MIGHT be connected, then uses optimization to find the exact edge weights
and directions — getting both speed and accuracy.

## How It Works

PC-NOTEARS combines the constraint-based PC algorithm's efficient skeleton discovery
with NOTEARS' continuous optimization for edge weight estimation and orientation.
Phase 1 runs PC-style CI tests to identify which variable pairs are connected.
Phase 2 runs a NOTEARS-like optimization restricted to the PC skeleton, estimating
edge weights while enforcing the acyclicity constraint h(W) = 0.

**Algorithm:**

- **PC skeleton phase:** Start with complete graph, remove edges via CI tests

at increasing conditioning set sizes

- **NOTEARS phase:** Initialize W from PC skeleton (OLS weights for connected pairs,

zero for removed pairs). Run gradient descent on L(W) + lambda * ||W||_1
subject to h(W) = tr(e^{W*W}) - d = 0, but only update entries in the skeleton

- Threshold small weights and return the DAG

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PCNOTEARSAlgorithm(CausalDiscoveryOptions)` | Initializes PC-NOTEARS with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverSkeleton(Matrix<>,Int32,Int32)` | Phase 1: PC-style skeleton discovery via conditional independence tests. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `OptimizeWeights(Matrix<>,Boolean[0:,0:],Int32,Int32)` | Phase 2: Optimize edge weights using coordinate descent restricted to skeleton. |

