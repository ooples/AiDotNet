---
title: "IQRDetector<T>"
description: "Detects anomalies using the Interquartile Range (IQR) method."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using the Interquartile Range (IQR) method.

## For Beginners

The IQR method is based on quartiles, which divide your data into four parts:

- Q1 (25th percentile): 25% of data is below this value
- Q3 (75th percentile): 75% of data is below this value
- IQR = Q3 - Q1: The range containing the middle 50% of data

Outliers are points below Q1 - k*IQR or above Q3 + k*IQR, where k is the multiplier (typically 1.5).

## How It Works

**When to use:**

- Your data may not be normally distributed
- You want a robust method that isn't affected by extreme outliers
- This is the same method used in box plots

**Industry Standard Defaults:**

- Multiplier: 1.5 (identifies "mild" outliers)
- Use 3.0 for "extreme" outliers only

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IQRDetector(Double,Double,Int32)` | Creates a new IQR-based anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IQR` | Gets the IQR values for each feature. |
| `LowerBounds` | Gets the lower bounds (Q1 - multiplier * IQR) for each feature. |
| `Multiplier` | Gets the IQR multiplier for determining outlier boundaries. |
| `Q1` | Gets the Q1 (25th percentile) values for each feature. |
| `Q3` | Gets the Q3 (75th percentile) values for each feature. |
| `UpperBounds` | Gets the upper bounds (Q3 + multiplier * IQR) for each feature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

