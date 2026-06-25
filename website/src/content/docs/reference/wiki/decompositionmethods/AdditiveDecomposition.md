---
title: "AdditiveDecomposition<T>"
description: "Implements additive time series decomposition, breaking a time series into trend, seasonal, and residual components."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Implements additive time series decomposition, breaking a time series into trend, seasonal, and residual components.

## For Beginners

Time series decomposition is like breaking down a complex signal (like sales data over time) 
into simpler parts that are easier to understand. The additive model assumes these components add up to form 
the original data: Original = Trend + Seasonal + Residual.

## How It Works

- Trend: The long-term progression (going up, down, or staying flat over time)
- Seasonal: Repeating patterns at fixed intervals (like higher sales every December)
- Residual: What's left over after removing trend and seasonal components (the "noise")

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdditiveDecomposition(Vector<>,AdditiveDecompositionAlgorithmType)` | Creates a new instance of the AdditiveDecomposition class |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateSeasonalExponentialSmoothing(Vector<>)` | Calculates the seasonal component using exponential smoothing |
| `CalculateSeasonalMovingAverage(Vector<>)` | Calculates the seasonal component using the moving average method |
| `CalculateTrendExponentialSmoothing` | Calculates the trend component using exponential smoothing |
| `CalculateTrendMovingAverage` | Calculates the trend component using a moving average method |
| `CycleSubseriesSmoothing(Vector<>,Int32)` | Performs cycle-subseries smoothing on the data |
| `Decompose` | Performs the decomposition based on the selected algorithm |
| `DecomposeExponentialSmoothing` | Decomposes the time series using the Exponential Smoothing method |
| `DecomposeMovingAverage` | Decomposes the time series using the Moving Average method |
| `DecomposeSTL` | Decomposes the time series using the Seasonal and Trend decomposition using Loess (STL) method |
| `LoessSmoothing(List<ValueTuple<,>>,Double)` | Applies LOESS smoothing to a list of (x,y) data points |
| `LoessSmoothing(Vector<>,Int32)` | Applies LOESS (Locally Estimated Scatterplot Smoothing) to a vector of data |
| `LowPassFilter(Vector<>,Int32)` | Applies a low-pass filter to the data |
| `PerformSTLDecomposition` | Performs the STL decomposition algorithm |
| `SubtractVectors(Vector<>,Vector<>)` | Subtracts one vector from another element by element |
| `TriCube()` | Calculates the tri-cube weight function used in LOESS smoothing |
| `WeightedLeastSquares(List<ValueTuple<,,>>)` | Calculates a weighted average of data points |

## Fields

| Field | Summary |
|:-----|:--------|
| `_algorithm` | The algorithm type used for decomposition |

