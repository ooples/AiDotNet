---
title: "ConstraintBasedBase<T>"
description: "Base class for constraint-based causal discovery algorithms (PC, FCI, MMPC, etc.)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalDiscovery.ConstraintBased`

Base class for constraint-based causal discovery algorithms (PC, FCI, MMPC, etc.).

## For Beginners

These algorithms work by asking "Are variables X and Y still related
after we account for other variables?" If not, there's no direct causal link between them.
By systematically testing all variable pairs, the algorithm builds a causal graph.

## How It Works

Constraint-based methods learn causal structure by performing conditional independence (CI) tests.
They start with a complete graph and remove edges between variables that are found to be
conditionally independent given some set of other variables.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Significance level (alpha) for conditional independence tests. |
| `Category` |  |
| `MaxConditioningSetSize` | Maximum size of conditioning sets to test. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyConstraintOptions(CausalDiscoveryOptions)` | Applies options from CausalDiscoveryOptions. |
| `ComputeCorrelation(Matrix<>,Int32,Int32)` | Computes Pearson correlation between two columns of data. |
| `ComputePartialCorr(Matrix<>,Int32,Int32,List<Int32>)` | Computes partial correlation between variables i and j given conditioning set. |
| `ComputeResiduals(Matrix<>,Int32,List<Int32>)` | Computes residuals of column target after regressing on predictor columns. |
| `GetCombinations(List<Int32>,Int32)` | Generates all combinations of size k from the given list. |
| `NormalCDF(Double)` | Standard normal CDF approximation. |
| `TestCI(Matrix<>,Int32,Int32,List<Int32>,Double)` | Tests conditional independence between variables i and j given conditioning set, using Fisher's z-transform of partial correlation. |

