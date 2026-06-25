---
title: "FeatureBaggingDetector<T>"
description: "Detects anomalies using Feature Bagging ensemble method."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Ensemble`

Detects anomalies using Feature Bagging ensemble method.

## For Beginners

Feature Bagging creates multiple anomaly detectors, each trained on
a random subset of features. The final score is the combination of all detector scores.
This helps handle high-dimensional data where different feature subsets may reveal different anomalies.

## How It Works

The algorithm works by:

1. Create n_estimators base detectors
2. Each detector uses a random subset of features
3. Train each detector on its feature subset
4. Combine scores using averaging or maximum

**When to use:**

- High-dimensional data
- When different feature combinations may reveal different anomalies
- When you want more robust detection than single-detector methods

**Industry Standard Defaults:**

- N estimators: 10
- Max features: 0.5 (50% of features per detector)
- Contamination: 0.1 (10%)

Reference: Lazarevic, A., Kumar, V. (2005). "Feature Bagging for Outlier Detection." KDD.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureBaggingDetector(Int32,Double,FeatureBaggingDetector<>.CombinationMethod,Double,Int32)` | Creates a new Feature Bagging anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxFeatures` | Gets the maximum features fraction. |
| `NEstimators` | Gets the number of estimators. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

