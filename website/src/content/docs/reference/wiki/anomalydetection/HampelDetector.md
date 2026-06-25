---
title: "HampelDetector<T>"
description: "Detects anomalies using Hampel identifier (median-based outlier detection)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using Hampel identifier (median-based outlier detection).

## For Beginners

The Hampel identifier is a robust outlier detection method that uses
a sliding window to compute local median and MAD. It's particularly effective for time series
data where local context matters.

## How It Works

The algorithm works by:

1. For each point, compute median and MAD in a local window
2. Compute the deviation from local median in MAD units
3. Points exceeding the threshold are anomalies

**When to use:**

- Time series with local patterns
- When global statistics don't capture local anomalies
- As a robust alternative to moving average methods

**Industry Standard Defaults:**

- Window size: 7 (3 on each side)
- Threshold: 3 MAD units
- Scale factor: 1.4826
- Contamination: 0.1 (10%)

Reference: Hampel, F.R. (1974). "The Influence Curve and its Role in Robust Estimation."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HampelDetector(Int32,Double,Double,Double,Int32)` | Creates a new Hampel anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ThresholdMAD` | Gets the threshold in MAD units. |
| `WindowSize` | Gets the window size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

