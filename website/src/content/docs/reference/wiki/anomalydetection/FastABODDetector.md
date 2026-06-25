---
title: "FastABODDetector<T>"
description: "Detects anomalies using Fast Angle-Based Outlier Detection (FastABOD)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.AngleBased`

Detects anomalies using Fast Angle-Based Outlier Detection (FastABOD).

## For Beginners

FastABOD is an optimized version of ABOD that uses only
k-nearest neighbors instead of all points. This makes it much faster while
maintaining good detection quality.

## How It Works

The algorithm works by:

1. For each point p, find k nearest neighbors
2. Compute angles only between neighbor pairs
3. Calculate angle variance as outlier score

**When to use:**

- Large datasets where full ABOD is too slow
- High-dimensional data
- When you need faster runtime with slight accuracy trade-off

**Industry Standard Defaults:**

- K (neighbors): 10
- Contamination: 0.1 (10%)

Reference: Kriegel, H., et al. (2008). "Angle-Based Outlier Detection in High-dimensional Data." KDD.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastABODDetector(Int32,Double,Int32)` | Creates a new FastABOD anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `K` | Gets the number of neighbors used for detection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

