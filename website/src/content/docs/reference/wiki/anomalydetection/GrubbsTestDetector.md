---
title: "GrubbsTestDetector<T>"
description: "Detects anomalies using Grubbs' Test for a single outlier."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Statistical`

Detects anomalies using Grubbs' Test for a single outlier.

## For Beginners

Grubbs' Test (also called the maximum normed residual test) identifies
a single outlier in a univariate dataset. It tests whether the most extreme value is
significantly different from the others.

## How It Works

The test statistic is: G = max(|x_i - mean|) / std
where the outlier is the point furthest from the mean relative to standard deviation.

**When to use:**

- When you suspect exactly one outlier in your data
- Data is approximately normally distributed
- Dataset is relatively small (n < 100)

**Industry Standard Defaults:**

- Alpha (significance level): 0.05 (5%)
- For multivariate data, applies test to each feature

Reference: Grubbs, F. E. (1950). "Sample criteria for testing outlying observations."
Annals of Mathematical Statistics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GrubbsTestDetector(Double,Double,Int32)` | Creates a new Grubbs' Test anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the significance level (alpha) for the test. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

