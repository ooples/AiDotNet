---
title: "ESDDetector<T>"
description: "Detects anomalies using ESD-based (Extreme Studentized Deviate) scoring."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using ESD-based (Extreme Studentized Deviate) scoring.

## For Beginners

This detector uses standardized residuals (the core statistic from the ESD test)
to identify outliers. Points that deviate significantly from the mean (measured in standard deviations)
are flagged as anomalies.

## How It Works

**Implementation:** This is a scoring-based approach following industry standards:

1. Compute the ESD statistic for each point: max|x - mean| / std across features
2. Use contamination-based threshold (top X% of scores are anomalies)
3. CriticalValue property available (computed from training sample size, valid for same n)

**When to use:**

- When you expect multiple outliers
- Data is approximately normally distributed
- You want Z-score style detection with contamination-based thresholding

**Industry Standard Defaults:**

- Alpha (significance level): 0.05 (5%) - used for critical value computation
- Contamination: 10% of data flagged as outliers

Reference: Rosner, B. (1983). "Percentage Points for a Generalized ESD Many-Outlier Procedure."
Technometrics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ESDDetector(Double,Nullable<Int32>,Double,Int32)` | Creates a new ESD anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the significance level (alpha) for the test. |
| `CriticalValue` | Gets the ESD critical value computed during fitting. |
| `MaxOutliers` | Gets the maximum number of outliers to detect. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

