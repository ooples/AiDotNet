---
title: "XGBODDetector<T>"
description: "Detects anomalies using XGBOD (Extreme Gradient Boosting Outlier Detection)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Ensemble`

Detects anomalies using XGBOD (Extreme Gradient Boosting Outlier Detection).

## For Beginners

XGBOD is a semi-supervised method that creates new features from
unsupervised outlier detection scores, then trains a supervised classifier on these
enhanced features. It combines the best of both worlds.

## How It Works

The algorithm works by:

1. Train multiple unsupervised outlier detectors
2. Generate outlier scores as new features (TOS: Transformed Outlier Scores)
3. Combine TOS with original features
4. Train gradient boosting classifier on enhanced features

**When to use:**

- When you have some labeled anomaly examples
- When feature engineering with outlier scores helps
- As a powerful ensemble method

**Industry Standard Defaults:**

- N estimators: 10 unsupervised detectors
- Boosting rounds: 100
- Contamination: 0.1 (10%)

Reference: Zhao, Y., Hryniewicki, M.K. (2018). "XGBOD: Improving Supervised Outlier
Detection with Unsupervised Representation Learning." IJCNN.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `XGBODDetector(Int32,Int32,Double,Int32)` | Creates a new XGBOD anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BoostingRounds` | Gets the number of boosting rounds. |
| `NEstimators` | Gets the number of base estimators. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

