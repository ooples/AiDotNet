---
title: "RandomSubspaceDetector<T>"
description: "Detects anomalies using Random Subspace ensemble method."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Ensemble`

Detects anomalies using Random Subspace ensemble method.

## For Beginners

Random Subspace creates multiple detectors, each trained on a randomly
selected subset of features. This helps detect anomalies that may only be visible in certain
feature subspaces, which is common in high-dimensional data.

## How It Works

The algorithm works by:

1. Generate n_estimators random feature subsets
2. Train a base detector on each subspace
3. Combine scores using averaging or voting

**When to use:**

- High-dimensional data
- When anomalies hide in subspaces
- For robust detection across feature combinations

**Industry Standard Defaults:**

- N estimators: 20
- Max features: sqrt(n_features)
- Combination: Average
- Contamination: 0.1 (10%)

Reference: Keller, F., Muller, E., Bohm, K. (2012). "HiCS: High Contrast Subspaces for
Density-Based Outlier Ranking." ICDE.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomSubspaceDetector(Int32,Int32,Double,Int32)` | Creates a new Random Subspace anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxFeatures` | Gets the maximum features per subspace. |
| `NEstimators` | Gets the number of estimators. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

