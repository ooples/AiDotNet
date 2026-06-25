---
title: "ChiSquareDetector<T>"
description: "Detects anomalies using the Chi-Square test for multivariate data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using the Chi-Square test for multivariate data.

## For Beginners

The Chi-Square (Mahalanobis distance) detector identifies outliers
by measuring how far a point is from the center of the data, accounting for correlations
between features. Points far from the center in Mahalanobis distance are anomalies.

## How It Works

The algorithm computes:
D^2 = (x - mean)' * Cov^(-1) * (x - mean)
This squared Mahalanobis distance follows a Chi-Square distribution with p degrees of freedom
(where p = number of features) under the assumption of multivariate normality.

**When to use:**

- Multivariate data (multiple correlated features)
- Data is approximately multivariate normal
- You want to account for correlations between features

**Industry Standard Defaults:**

- Alpha (significance level): 0.05 (5%)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChiSquareDetector(Double,Double,Int32)` | Creates a new Chi-Square anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the significance level (alpha) for the test. |
| `ChiSquareCritical` | Gets the Chi-Square critical value based on the fitted degrees of freedom (number of features) and alpha. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `Predict(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

