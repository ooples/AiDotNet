---
title: "ARIMADetector<T>"
description: "Detects anomalies in time series using ARIMA model residuals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TimeSeries`

Detects anomalies in time series using ARIMA model residuals.

## For Beginners

ARIMA (AutoRegressive Integrated Moving Average) is a classic
time series model. It predicts future values based on past values and errors. Points
where the prediction error is large are flagged as anomalies.

## How It Works

The algorithm works by:

1. Fit an ARIMA(p,d,q) model to the time series
2. Compute prediction residuals
3. Points with large residuals (standardized) are anomalies

**When to use:**

- Stationary time series (or made stationary via differencing)
- Detecting point anomalies that don't fit the temporal pattern
- Well-understood temporal dependencies

**Industry Standard Defaults:**

- p (AR order): 2
- d (differencing): 1
- q (MA order): 2
- Contamination: 0.1 (10%)

Reference: Box, G.E.P., Jenkins, G.M. (1970). "Time Series Analysis: Forecasting and Control."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ARIMADetector(Int32,Int32,Int32,Double,Int32)` | Creates a new ARIMA anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `D` | Gets the differencing order (d). |
| `P` | Gets the AR order (p). |
| `Q` | Gets the MA order (q). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

