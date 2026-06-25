---
title: "PercentileDetector<T>"
description: "Detects anomalies using percentile-based thresholds."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using percentile-based thresholds.

## For Beginners

This detector uses percentiles to identify extreme values. Points
that fall below the low percentile or above the high percentile are flagged as anomalies.
It's simple and effective for univariate or feature-wise anomaly detection.

## How It Works

The algorithm works by:

1. Compute the specified percentiles for each feature
2. For each point, check if values are outside the percentile bounds
3. Score based on how far outside the bounds a point falls

**When to use:**

- Simple univariate anomaly detection
- When you don't want to assume a distribution
- As a baseline method

**Industry Standard Defaults:**

- Low percentile: 5 (5th percentile)
- High percentile: 95 (95th percentile)
- Contamination: 0.1 (10%)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PercentileDetector(Double,Double,Double,Int32)` | Creates a new Percentile anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HighPercentile` | Gets the high percentile threshold. |
| `LowPercentile` | Gets the low percentile threshold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

