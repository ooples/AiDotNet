---
title: "MovingAverageDetector<T>"
description: "Detects anomalies using moving average deviation in time series."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.TimeSeries`

Detects anomalies using moving average deviation in time series.

## For Beginners

This detector computes a moving average and identifies points that
deviate significantly from their local average. It's one of the simplest and most intuitive
time series anomaly detection methods.

## How It Works

The algorithm works by:

1. Compute moving average and moving standard deviation
2. For each point, compute deviation from local mean in std units
3. High deviations indicate anomalies

**When to use:**

- Simple time series without complex patterns
- As a baseline method
- Real-time anomaly detection (streaming data)

**Industry Standard Defaults:**

- Window size: 20
- Threshold: 3 standard deviations
- Contamination: 0.1 (10%)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MovingAverageDetector(Int32,Double,Double,Int32)` | Creates a new Moving Average anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `StdThreshold` | Gets the standard deviation threshold. |
| `WindowSize` | Gets the window size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

