---
title: "PCADetector<T>"
description: "Detects anomalies using Principal Component Analysis (PCA) reconstruction error."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Linear`

Detects anomalies using Principal Component Analysis (PCA) reconstruction error.

## For Beginners

PCA-based anomaly detection works by projecting data onto
the main directions of variation (principal components) and measuring how well
points can be reconstructed. Anomalies have high reconstruction error.

## How It Works

The algorithm works by:

1. Fit PCA on training data to find principal components
2. Project each point onto these components
3. Reconstruct the point from the projection
4. Measure reconstruction error (anomaly score)

**When to use:**

- Linear relationships in data
- When anomalies deviate from the main data structure
- High-dimensional data that can be compressed

**Industry Standard Defaults:**

- Components: min(n_samples, n_features) or specified
- Variance threshold: 0.95 (keep 95% of variance)
- Contamination: 0.1 (10%)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PCADetector(Nullable<Int32>,Double,Double,Int32)` | Creates a new PCA anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NComponents` | Gets the number of components used. |
| `VarianceThreshold` | Gets the variance threshold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `EigenvalueFloor` | Eigenvalues below this threshold are treated as zero to avoid division by near-zero values in Mahalanobis distance. |

