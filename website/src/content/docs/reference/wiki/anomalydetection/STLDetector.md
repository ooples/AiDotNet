---
title: "STLDetector<T>"
description: "Detects anomalies using STL (Seasonal and Trend decomposition using Loess)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TimeSeries`

Detects anomalies using STL (Seasonal and Trend decomposition using Loess).

## For Beginners

STL decomposes a time series into three components: trend, seasonal,
and residual. The residual component contains the irregular variations. Large residuals
indicate anomalies that don't fit the expected trend and seasonal patterns.

## How It Works

The algorithm works by:

1. Apply STL decomposition to extract trend, seasonal, and residual
2. Standardize the residual component
3. Large standardized residuals indicate anomalies

**When to use:**

- Time series with clear trend and/or seasonal patterns
- When anomalies disrupt expected patterns
- Sales, weather, sensor data with periodicity

**Industry Standard Defaults:**

- Season length: 7 (weekly pattern)
- Trend smoothness: 15
- Contamination: 0.1 (10%)

Reference: Cleveland, R.B., et al. (1990). "STL: A Seasonal-Trend Decomposition Procedure
Based on Loess." Journal of Official Statistics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `STLDetector(Int32,Int32,Double,Int32)` | Creates a new STL anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SeasonLength` | Gets the season length. |
| `TrendSmoothness` | Gets the trend smoothness parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

