---
title: "HybridBase<T>"
description: "Base class for hybrid causal discovery algorithms that combine constraint-based and score-based methods."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalDiscovery.Hybrid`

Base class for hybrid causal discovery algorithms that combine constraint-based and score-based methods.

## For Beginners

Hybrid algorithms get the best of both worlds. Constraint-based methods
are good at ruling out edges quickly, while score-based methods are good at finding the best
structure among remaining options. Combining them gives faster AND more accurate results.

## How It Works

Hybrid methods first use constraint-based tests to restrict the search space (e.g., finding
candidate parents via conditional independence tests), then use score-based search to find the
optimal DAG within the restricted space.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Significance level for constraint-based phase. |
| `Category` |  |
| `MaxParents` | Maximum number of parents per variable (restricts search space). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyHybridOptions(CausalDiscoveryOptions)` | Applies options from CausalDiscoveryOptions. |
| `ComputeBIC(Matrix<>,Int32,List<Int32>)` | Computes BIC score for a variable given its parents. |
| `ComputeCorrelation(Matrix<>,Int32,Int32)` | Computes Pearson correlation between two columns of data. |

