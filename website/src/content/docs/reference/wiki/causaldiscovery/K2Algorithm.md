---
title: "K2Algorithm<T>"
description: "K2 Algorithm — score-based learning with known variable ordering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ScoreBased`

K2 Algorithm — score-based learning with known variable ordering.

## For Beginners

If you already know the rough order in which variables cause
each other (e.g., age before income before spending), K2 efficiently finds the
exact connections. It's very fast but requires this ordering as input. When no
ordering is provided, a heuristic based on correlation structure is used.

## How It Works

K2 learns a Bayesian network structure given a known topological ordering of variables.
For each variable in order, it greedily adds parents from variables earlier in the ordering
that maximize the BIC score, up to a maximum number of parents.

**Algorithm:**

- Determine a variable ordering (using correlation-based heuristic if not provided)
- For each variable X_i in order:
- Initialize parents(X_i) = empty
- Repeat: find the predecessor z that maximizes BIC(X_i | parents + z)
- Add z if BIC improves and |parents| < maxParents
- Stop when no improvement or max parents reached

Reference: Cooper and Herskovits (1992), "A Bayesian Method for the Induction
of Probabilistic Networks from Data".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `K2Algorithm(CausalDiscoveryOptions)` | Initializes K2 with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetermineOrdering(Matrix<>,Int32)` | Determines a variable ordering using a correlation-based heuristic. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `EstimateEdgeWeight(Matrix<>,Int32,Int32,HashSet<Int32>)` | Estimates the partial regression coefficient for parent→child, controlling for all other parents. |

