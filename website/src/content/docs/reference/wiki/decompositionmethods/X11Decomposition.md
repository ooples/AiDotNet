---
title: "X11Decomposition<T>"
description: "Implements the X-11 method for time series decomposition, which breaks down a time series into trend, seasonal, and irregular components."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Implements the X-11 method for time series decomposition, which breaks down a time series into trend, seasonal, and irregular components.

## For Beginners

The X-11 method is a statistical technique that helps understand patterns in data that changes over time.
It separates your data into three main parts:

- Trend: The long-term direction of your data (going up, down, or staying flat over time)
- Seasonal: Regular patterns that repeat at fixed intervals (like higher sales during holidays)
- Irregular: Random fluctuations that don't follow any pattern

This helps you understand what's really happening in your data by removing predictable patterns.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `X11Decomposition(Vector<>,Int32,Int32,X11AlgorithmType)` | Creates a new instance of the X11Decomposition class and performs the decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyExp(Vector<>)` | Applies the exponential function to each element in a vector. |
| `CalculateHendersonWeights(Int32)` | Calculates the weights for the Henderson moving average. |
| `CenteredMovingAverage(Vector<>,Int32)` | Calculates a centered moving average of the input data. |
| `Decompose` | Performs the time series decomposition based on the selected algorithm type. |
| `DecomposeLogAdditive` | Performs log-additive decomposition of the time series. |
| `DecomposeMultiplicative` | Performs multiplicative X-11 decomposition. |
| `DecomposeStandard` | Performs standard additive X-11 decomposition. |
| `EnsureMultiplicativeConsistency` | Ensures that the product of all components equals the original time series. |
| `EstimateSeasonalFactors(Vector<>)` | Estimates seasonal factors for additive decomposition. |
| `EstimateSeasonalFactorsMultiplicative(Vector<>)` | Estimates seasonal factors for multiplicative decomposition. |
| `HendersonMovingAverage(Vector<>,Int32)` | Applies the Henderson moving average to smooth a time series. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_algorithmType` | The type of X-11 algorithm to use for decomposition. |
| `_seasonalPeriod` | The number of observations in one complete seasonal cycle. |
| `_trendCycleMovingAverageWindow` | The window size used for the moving average calculation of the trend component. |

