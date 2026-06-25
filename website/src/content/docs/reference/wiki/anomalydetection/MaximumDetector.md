---
title: "MaximumDetector<T>"
description: "Combines multiple anomaly detectors using maximum score strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Ensemble`

Combines multiple anomaly detectors using maximum score strategy.

## For Beginners

This ensemble method takes the maximum anomaly score across all base
detectors. If ANY detector thinks a point is anomalous, it gets a high score. This is
useful when you want to catch anomalies that only specific detectors can find.

## How It Works

The algorithm works by:

1. Train each base detector on the data
2. Normalize scores from each detector to [0,1]
3. Take the maximum normalized score for each point
4. Points with high max score are anomalies

**When to use:**

- When different anomaly types need different detectors
- When you want high recall (catch most anomalies)
- When false negatives are costly

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaximumDetector(Double,Int32)` | Creates a new Maximum ensemble anomaly detector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

