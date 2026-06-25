---
title: "SeasonalHybridESDDetector<T>"
description: "Detects anomalies in time series data using Seasonal Hybrid ESD (S-H-ESD)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TimeSeries`

Detects anomalies in time series data using Seasonal Hybrid ESD (S-H-ESD).

## For Beginners

S-H-ESD combines seasonal decomposition with the Generalized ESD test
to detect anomalies in time series data. It handles seasonality by removing the seasonal
pattern before testing for outliers, making it effective for data with daily, weekly, or
yearly patterns.

## How It Works

The algorithm works by:

1. Decompose the time series using STL (Seasonal and Trend decomposition using Loess)
2. Extract the residual component (data after removing trend and seasonality)
3. Apply GESD test on the residuals to find anomalies

**When to use:**

- Time series data with seasonal patterns
- Detecting anomalies that deviate from expected seasonal behavior
- Server metrics, website traffic, sensor data with regular patterns

**Industry Standard Defaults:**

- Season length: 7 (weekly pattern)
- Alpha: 0.05 (5% significance level)
- Max anomalies: 10% of data
- Contamination: 0.1 (10%)

Reference: Twitter's "AnomalyDetection" package, based on Rosner (1983) GESD test.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SeasonalHybridESDDetector(Int32,Double,Nullable<Int32>,Double,Int32)` | Creates a new Seasonal Hybrid ESD anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the significance level. |
| `SeasonLength` | Gets the season length (period). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

