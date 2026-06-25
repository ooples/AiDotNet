---
title: "ABODDetector<T>"
description: "Detects anomalies using Angle-Based Outlier Detection (ABOD)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.AngleBased`

Detects anomalies using Angle-Based Outlier Detection (ABOD).

## For Beginners

ABOD detects outliers by analyzing the angles formed between
a point and all pairs of other points. Outliers tend to have angles that point in
similar directions (low variance), while inliers have diverse angle patterns.

## How It Works

The algorithm works by:

1. For each point p, compute angles between p and all pairs of other points
2. Calculate the variance of these angles (weighted by distance)
3. Low variance indicates the point is an outlier

**When to use:**

- High-dimensional data (works better than distance-based methods)
- When the "curse of dimensionality" affects other methods
- Data has complex structure

**Industry Standard Defaults:**

- Contamination: 0.1 (10%)
- For large datasets, use FastABOD variant

Reference: Kriegel, H., et al. (2008). "Angle-Based Outlier Detection in High-dimensional Data." KDD.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ABODDetector(Double,Int32)` | Creates a new ABOD anomaly detector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

