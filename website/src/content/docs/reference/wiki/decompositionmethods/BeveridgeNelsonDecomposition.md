---
title: "BeveridgeNelsonDecomposition<T>"
description: "Implements the Beveridge-Nelson decomposition method for time series analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DecompositionMethods.TimeSeriesDecomposition`

Implements the Beveridge-Nelson decomposition method for time series analysis.

## How It Works

**For Beginners:** The Beveridge-Nelson decomposition separates a time series into two components:

1. A permanent component (trend) - the long-term path the data would follow if there were no temporary fluctuations
2. A temporary component (cycle) - short-term fluctuations that eventually fade away

This is useful for understanding which changes in your data (like stock prices or economic indicators)
are likely to persist versus which are temporary fluctuations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BeveridgeNelsonDecomposition(Vector<>,BeveridgeNelsonAlgorithmType,ARIMAOptions<>,Int32,Matrix<>)` | Initializes a new instance of the Beveridge-Nelson decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateARIMACycle(Vector<>)` | Calculates the cyclical component using the ARIMA-based Beveridge-Nelson method. |
| `CalculateARIMATrend(ARIMAModel<>)` | Calculates the trend component using an ARIMA model-based Beveridge-Nelson method. |
| `CalculateLongRunImpactMatrix(VectorAutoRegressionModel<>,VARModelOptions<>)` | Calculates the long-run impact matrix for multivariate decomposition. |
| `CalculateStandardCycle(Vector<>)` | Calculates the cyclical component using the standard Beveridge-Nelson method. |
| `CalculateStandardTrend` | Calculates the trend component using the standard Beveridge-Nelson method. |
| `Decompose` | Performs the decomposition based on the selected algorithm. |
| `DecomposeARIMA` | Performs the ARIMA-based Beveridge-Nelson decomposition. |
| `DecomposeMultivariate` | Performs the multivariate Beveridge-Nelson decomposition. |
| `DecomposeStandard` | Performs the standard Beveridge-Nelson decomposition. |

