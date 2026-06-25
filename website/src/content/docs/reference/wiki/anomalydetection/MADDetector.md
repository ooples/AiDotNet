---
title: "MADDetector<T>"
description: "Detects anomalies using Median Absolute Deviation (MAD)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using Median Absolute Deviation (MAD).

## For Beginners

MAD is a robust measure of spread that uses the median instead of
the mean. Unlike standard deviation, MAD is resistant to outliers. Points far from the
median (in terms of MAD units) are flagged as anomalies.

## How It Works

The algorithm works by:

1. Compute the median of each feature
2. Compute MAD = median(|x - median|) for each feature
3. Score points based on their deviation from median in MAD units
4. High scores indicate anomalies

**When to use:**

- Data with heavy-tailed distributions
- When outliers may skew standard statistics
- As a robust alternative to Z-score

**Industry Standard Defaults:**

- Threshold: 3.5 MAD units
- Scale factor: 1.4826 (for Gaussian consistency)
- Contamination: 0.1 (10%)

Reference: Leys, C., et al. (2013). "Detecting outliers: Do not use standard deviation
around the mean, use absolute deviation around the median." Journal of Experimental Social Psychology.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MADDetector(Double,Double,Double,Int32)` | Creates a new MAD anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MADThreshold` | Gets the MAD threshold for anomaly detection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

