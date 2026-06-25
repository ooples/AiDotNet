---
title: "IGCIAlgorithm<T>"
description: "IGCI (Information-Geometric Causal Inference) — bivariate causal discovery via entropy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Functional`

IGCI (Information-Geometric Causal Inference) — bivariate causal discovery via entropy.

## For Beginners

IGCI works by checking whether the relationship between two
variables is "smoother" in one direction than the other. If X causes Y, the mapping
from X to Y tends to have slopes that anti-correlate with the density of X — a
natural geometric property of cause-effect relationships. This method is very fast
(pairwise comparisons only) but only works for bivariate, deterministic-ish relationships.

## How It Works

IGCI determines causal direction between pairs of variables by exploiting the
information-geometric principle: if X causes Y, the marginal distribution of X
and the conditional P(Y|X) are independent, which means the slope of Y=f(X)
tends to correlate negatively with the density of X at that point.

**Algorithm:**

- For each pair (X, Y), uniformize both variables to [0,1] via rank transform
- Sort by X values and compute the average log-slope: sum(log|dy/dx|)/n
- The causal direction score is: score = mean(log|f'(x)|)
- If score < 0, X→Y is preferred; if score > 0, Y→X is preferred
- Build adjacency matrix from pairwise decisions

Reference: Janzing et al. (2012), "Information-Geometric Approach to Inferring
Causal Directions", Artificial Intelligence.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IGCIAlgorithm(CausalDiscoveryOptions)` | Initializes IGCI with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeIGCIScore(Matrix<>,Int32,Int32,Int32)` | Computes the IGCI score for a pair of variables. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `MinMaxRescale(Double[])` | Affinely rescales a vector to [0,1] (the IGCI uniform reference measure: the variable's RANGE is normalized, preserving the shape of the function between the variables, unlike a rank transform which destroys it). |

