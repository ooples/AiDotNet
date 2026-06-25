---
title: "PCMCIAlgorithm<T>"
description: "PCMCI — PC algorithm for Momentary Conditional Independence in time series."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.TimeSeries`

PCMCI — PC algorithm for Momentary Conditional Independence in time series.

## For Beginners

PCMCI is designed for time series where you want to know which
variables' past values help predict other variables' current values. It's more reliable
than simple Granger causality because it conditions on the right set of variables.

## How It Works

PCMCI combines a condition-selection step (based on PC's skeleton discovery) with
momentary conditional independence (MCI) tests. It first identifies the relevant
lagged parents of each variable, then tests for direct causal links conditioned
on those parents.

**Algorithm:**

- PC₁: For each variable, identify candidate lagged parents using iterative CI tests
- MCI: Test X(t-τ) → Y(t) conditioned on parents of both X and Y

Reference: Runge et al. (2019), "Detecting and Quantifying Causal Associations in
Large Nonlinear Time Series Datasets", Science Advances.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLaggedPartialCorrelation(Matrix<>,Int32,Int32,Int32,HashSet<ValueTuple<Int32,Int32>>,Int32)` | Computes partial correlation between target(t) and source(t-lag) conditioned on a set of lagged parent variables, using OLS residualization. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `FisherZTestPValue(Double,Int32,Int32)` | Computes a two-sided p-value from a partial correlation using the Fisher z-transform. |
| `NormalCdfComplement(Double)` | Computes 1 - Φ(x) for the standard normal distribution using the Abramowitz and Stegun approximation. |
| `OLSResiduals(Matrix<>,Vector<>,Int32,Int32)` | Computes OLS residuals: y - Z * (Z'Z)^{-1} Z'y using the normal equations. |
| `PearsonCorrelation(Vector<>,Vector<>,Int32)` | Computes Pearson correlation between two vectors. |

