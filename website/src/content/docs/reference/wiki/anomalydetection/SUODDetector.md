---
title: "SUODDetector<T>"
description: "Detects anomalies using SUOD (Scalable Unsupervised Outlier Detection)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Ensemble`

Detects anomalies using SUOD (Scalable Unsupervised Outlier Detection).

## For Beginners

SUOD is an acceleration framework that combines multiple
anomaly detection algorithms efficiently. It uses approximation techniques to
speed up detection while maintaining accuracy.

## How It Works

The algorithm works by:

1. Train multiple diverse base detectors
2. Use random projection for dimensionality reduction
3. Combine scores using robust averaging

**When to use:**

- Large datasets where speed matters
- When you want ensemble benefits with good performance
- As a general-purpose robust detector

**Industry Standard Defaults:**

- Base detectors: LOF, k-NN, Isolation Forest
- Contamination: 0.1 (10%)

Reference: Zhao, Y., et al. (2021). "SUOD: Accelerating Large-Scale Unsupervised
Heterogeneous Outlier Detection." MLSys.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SUODDetector(Boolean,Int32,Double,Int32)` | Creates a new SUOD anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `UseRandomProjection` | Gets whether random projection is used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

