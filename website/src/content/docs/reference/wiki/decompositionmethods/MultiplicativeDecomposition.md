---
title: "MultiplicativeDecomposition<T>"
description: "Performs multiplicative decomposition of time series data into trend, seasonal, and residual components."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Performs multiplicative decomposition of time series data into trend, seasonal, and residual components.

## How It Works

**For Beginners:** Multiplicative decomposition is used when the seasonal variations in your data increase 
or decrease proportionally with the level of the time series. In this model, the components are multiplied 
together (Original = Trend × Seasonal × Residual) rather than added. This is often appropriate for economic 
or financial data where percentage changes are more meaningful than absolute changes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiplicativeDecomposition(Vector<>,MultiplicativeAlgorithmType,Int32)` | Initializes a new instance of the `MultiplicativeDecomposition` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decompose` | Performs the time series decomposition using the selected algorithm. |
| `DecomposeGeometricMovingAverage` | Decomposes the time series using a geometric moving average approach. |
| `DecomposeLogTransformedSTL` | Decomposes the time series using a log-transformed STL (Seasonal-Trend decomposition using LOESS) approach. |
| `DecomposeMultiplicativeExponentialSmoothing` | Decomposes the time series using multiplicative exponential smoothing. |

