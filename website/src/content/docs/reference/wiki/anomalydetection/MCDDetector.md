---
title: "MCDDetector<T>"
description: "Detects anomalies using Minimum Covariance Determinant (MCD)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Linear`

Detects anomalies using Minimum Covariance Determinant (MCD).

## For Beginners

MCD is a robust method for estimating the center and spread of data.
Unlike standard mean and covariance, MCD is resistant to outliers by finding the subset
of points that minimizes the covariance determinant. Points far from this robust estimate
are anomalies.

## How It Works

The algorithm works by:

1. Find the subset of h points with minimum covariance determinant
2. Compute robust mean and covariance from this subset
3. Compute Mahalanobis distances using robust estimates
4. Points with large distances are anomalies

**When to use:**

- Data with outliers that corrupt standard statistics
- When you need robust location/scatter estimates
- Multivariate anomaly detection

**Industry Standard Defaults:**

- Support fraction: 0.5 (use 50% of data for robust estimate)
- Contamination: 0.1 (10%)

Reference: Rousseeuw, P.J. (1984). "Least Median of Squares Regression."
Rousseeuw, P.J., Driessen, K.V. (1999). "A Fast Algorithm for the Minimum Covariance Determinant Estimator."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MCDDetector(Double,Double,Int32)` | Creates a new MCD anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportFraction` | Gets the support fraction. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

