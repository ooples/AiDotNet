---
title: "ZScoreDetector<T>"
description: "Detects anomalies using the Z-Score method (standard score)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using the Z-Score method (standard score).

## For Beginners

The Z-Score measures how many standard deviations a value is from the mean.
A Z-Score of 0 means the value equals the mean. A Z-Score of 2 means the value is 2 standard
deviations above the mean. Points with extreme Z-Scores (typically |Z| > 3) are anomalies.

## How It Works

**When to use:**

- Your data is approximately normally distributed
- You want a simple, interpretable method
- You have single-variable or multi-variable data where anomalies are extreme in at least one dimension

**Industry Standard Defaults:**

- Threshold: 3.0 (flags ~0.3% of normally distributed data)
- Contamination: 0.1 (10%) - used for automatic threshold tuning

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ZScoreDetector(Double,Double,Int32)` | Creates a new Z-Score anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Means` | Gets the fitted mean values for each feature. |
| `StandardDeviations` | Gets the fitted standard deviation values for each feature. |
| `ZThreshold` | Gets the Z-Score threshold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `Predict(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

