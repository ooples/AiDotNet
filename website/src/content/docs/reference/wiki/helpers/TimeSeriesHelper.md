---
title: "TimeSeriesHelper<T>"
description: "Provides helper methods for time series analysis and forecasting."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides helper methods for time series analysis and forecasting.

## How It Works

**For Beginners:** Time series analysis is a technique used to analyze data points collected over time
to identify patterns and predict future values. This is commonly used for forecasting trends like
stock prices, weather patterns, or sales data.

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateARResiduals(Vector<>,Vector<>)` | Calculates the residuals (errors) of an Autoregressive (AR) model. |
| `CalculateAutoCorrelation(Vector<>,Int32)` | Calculates the autocorrelation of a time series at a specific lag. |
| `CalculateMultipleAutoCorrelation(Vector<>,Int32)` | Calculates the autocorrelation function (ACF) of a time series for lags 0 through maxLag. |
| `DifferenceSeries(Vector<>,Int32)` | Computes the differences between consecutive values in a time series. |
| `EnforceCoefficientStability(Vector<>,Double,Double)` | Enforces coefficient stability by sanitizing NaN/Infinity values and scaling coefficients when their absolute sum approaches or exceeds 1. |
| `EstimateARCoefficients(Vector<>,Int32,MatrixDecompositionType)` | Estimates the coefficients for an Autoregressive (AR) model. |
| `EstimateMACoefficients(Vector<>,Int32)` | Estimates the coefficients for a Moving Average (MA) model. |

