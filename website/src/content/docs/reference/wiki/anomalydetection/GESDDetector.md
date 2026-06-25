---
title: "GESDDetector<T>"
description: "Detects anomalies using GESD-based (Generalized Extreme Studentized Deviate) scoring."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using GESD-based (Generalized Extreme Studentized Deviate) scoring.

## For Beginners

This detector uses standardized residuals (the core statistic from the GESD test)
to identify outliers. Points that deviate significantly from the mean are flagged as anomalies.

## How It Works

**Implementation:** This is a scoring-based approach:

1. Compute the GESD statistic for each point: max|x - mean| / std across features
2. Use contamination-based threshold to classify anomalies (top X% of scores)

Critical values are computed from alpha for reference but thresholding uses contamination.

**When to use:**

- Data is approximately normally distributed
- You want Z-score style detection with contamination-based thresholding

**Industry Standard Defaults:**

- Alpha (significance level): 0.05 (5%)
- Contamination: 10% of data flagged as outliers

Reference: Rosner, B. (1983). "Percentage Points for a Generalized ESD Many-Outlier Procedure."
Technometrics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GESDDetector(Int32,Double,Double,Int32)` | Creates a new GESD anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the significance level (alpha) for the test. |
| `MaxOutliersCount` | Gets the maximum number of outliers to detect. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLambda(Int32,Double)` | Computes the critical value lambda for the GESD test. |
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

