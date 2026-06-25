---
title: "ECODDetector<T>"
description: "Detects anomalies using ECOD (Empirical Cumulative Distribution Functions for Outlier Detection)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Probabilistic`

Detects anomalies using ECOD (Empirical Cumulative Distribution Functions for Outlier Detection).

## For Beginners

ECOD uses the cumulative distribution function (CDF) to identify
outliers. Points with extreme values in any dimension have low probability under the
empirical distribution and are flagged as anomalies.

## How It Works

The algorithm works by:

1. Compute empirical CDF for each feature
2. Calculate tail probabilities for each point
3. Combine probabilities across features
4. Points with very low probabilities are anomalies

**When to use:**

- Large datasets (linear complexity O(n))
- High-dimensional data
- When you need a fast, parameter-free method

**Industry Standard Defaults:**

- No parameters to tune (parameter-free)
- Contamination: 0.1 (10%)

Reference: Li, Z., et al. (2022). "ECOD: Unsupervised Outlier Detection Using
Empirical Cumulative Distribution Functions." IEEE TKDE.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ECODDetector(Double,Int32)` | Creates a new ECOD anomaly detector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

