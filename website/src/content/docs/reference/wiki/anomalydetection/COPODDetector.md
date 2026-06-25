---
title: "COPODDetector<T>"
description: "Detects anomalies using Copula-Based Outlier Detection (COPOD)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Probabilistic`

Detects anomalies using Copula-Based Outlier Detection (COPOD).

## For Beginners

COPOD uses copulas (statistical functions that describe dependencies
between variables) to model the joint probability of data points. Points with very low
joint probability are anomalies - they are statistically unlikely given the data distribution.

## How It Works

The algorithm works by:

1. Transform each feature to empirical probability using ECDF
2. Model the joint distribution using empirical copula
3. Compute the negative log probability as the anomaly score

**When to use:**

- When features have different marginal distributions
- When you want a parameter-free method
- High-dimensional data with complex dependencies

**Industry Standard Defaults:**

- Contamination: 0.1 (10%)
- No hyperparameters to tune (parameter-free method)

Reference: Li, Z., et al. (2020). "COPOD: Copula-Based Outlier Detection." ICDM.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `COPODDetector(Double,Int32)` | Creates a new COPOD anomaly detector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

