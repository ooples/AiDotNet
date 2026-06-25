---
title: "EllipticEnvelopeDetector<T>"
description: "Detects anomalies using Elliptic Envelope (robust covariance estimation)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Linear`

Detects anomalies using Elliptic Envelope (robust covariance estimation).

## For Beginners

Elliptic Envelope fits an ellipse (in 2D) or ellipsoid (in higher dimensions)
around the data using robust estimation. Points far from this envelope are anomalies.
It's like drawing the smallest ellipse that contains most of the data.

## How It Works

The algorithm works by:

1. Estimate robust mean and covariance using Minimum Covariance Determinant (MCD)
2. Compute Mahalanobis distance for each point
3. Points with large distances are anomalies

**When to use:**

- Data is approximately Gaussian/elliptical
- You need robustness against outliers in the training data
- Multivariate anomaly detection

**Industry Standard Defaults:**

- Support fraction: 0.5 (use 50% of data for robust estimation)
- Contamination: 0.1 (10%)

Reference: Rousseeuw, P.J., Van Driessen, K. (1999). "A Fast Algorithm for the Minimum
Covariance Determinant Estimator." Technometrics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EllipticEnvelopeDetector(Double,Double,Int32)` | Creates a new Elliptic Envelope anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportFraction` | Gets the support fraction (proportion of data for robust estimation). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

