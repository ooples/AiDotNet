---
title: "DixonQTestDetector<T>"
description: "Detects anomalies using Dixon's Q Test for small datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using Dixon's Q Test for small datasets.

## For Beginners

Dixon's Q Test is designed for detecting a single outlier in small datasets
(typically n < 25). It compares the gap between the suspect value and its nearest neighbor
to the range of the entire dataset.

## How It Works

The test statistic is: Q = gap / range
where gap = |suspect value - nearest value| and range = max - min.

**When to use:**

- Small datasets (n < 25, ideally 3-10 samples)
- You suspect exactly one outlier
- Data is approximately normally distributed

**Industry Standard Defaults:**

- Alpha (significance level): 0.05 (5%)

Reference: Dixon, W. J. (1950). "Analysis of Extreme Values."
Annals of Mathematical Statistics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DixonQTestDetector(Double,Double,Int32)` | Creates a new Dixon's Q Test anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the significance level (alpha) for the test. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `GetQCritical(Int32)` | Gets the critical Q value for Dixon's test based on sample size and alpha level. |
| `Predict(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

