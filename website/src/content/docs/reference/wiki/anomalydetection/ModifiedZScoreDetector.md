---
title: "ModifiedZScoreDetector<T>"
description: "Detects anomalies using the Modified Z-Score method based on Median Absolute Deviation (MAD)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using the Modified Z-Score method based on Median Absolute Deviation (MAD).

## For Beginners

The Modified Z-Score is a robust alternative to the standard Z-Score.
Instead of using mean and standard deviation (which are sensitive to outliers), it uses:

- Median: The middle value when data is sorted
- MAD: Median Absolute Deviation, a robust measure of spread

Modified Z-Score = 0.6745 * (x - median) / MAD
(0.6745 is a scaling factor to make MAD comparable to standard deviation for normal distributions)

## How It Works

**When to use:**

- Your data contains extreme outliers that would skew mean/std
- Your data is not normally distributed
- You want a method that remains reliable even with 50% outliers

**Industry Standard Defaults:**

- Threshold: 3.5 (Iglewicz and Hoaglin recommendation)
- Alternative: 3.0 for more sensitivity

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModifiedZScoreDetector(Double,Double,Int32)` | Creates a new Modified Z-Score (MAD-based) anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MADs` | Gets the fitted MAD (Median Absolute Deviation) values for each feature. |
| `Medians` | Gets the fitted median values for each feature. |
| `ModifiedZThreshold` | Gets the Modified Z-Score threshold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `Predict(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `MAD_SCALE_FACTOR` | Scaling factor to make MAD comparable to standard deviation for normal distributions. |

