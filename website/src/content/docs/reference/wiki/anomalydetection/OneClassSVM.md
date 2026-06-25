---
title: "OneClassSVM<T>"
description: "Implements One-Class SVM for novelty/outlier detection using the RBF kernel."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.Linear`

Implements One-Class SVM for novelty/outlier detection using the RBF kernel.

## For Beginners

One-Class SVM learns a "boundary" around your normal data points.
New points that fall outside this boundary are considered outliers. It's like drawing
a flexible shape around your data - anything outside the shape is unusual.

## How It Works

The algorithm works by:

1. Mapping data to a high-dimensional space using the RBF (Radial Basis Function) kernel
2. Finding a hyperplane that separates most data points from the origin
3. Points far from this hyperplane (on the wrong side) are outliers

**When to use:** One-Class SVM is particularly effective for:

- Novelty detection (training only on "normal" data)
- When you have a clear notion of what "normal" looks like
- Data with complex, non-linear boundaries

**Industry Standard Defaults:**

- Nu: 0.1 (upper bound on outlier fraction)
- Gamma: Auto-detect (1/n_features)
- Max iterations: 1000
- Tolerance: 1e-3

Reference: Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., and Williamson, R. C. (2001).
"Estimating the Support of a High-Dimensional Distribution." Neural Computation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OneClassSVM(Double,Double,Int32,Double,Double,Int32)` | Creates a new One-Class SVM anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Gamma` | Gets the gamma parameter for the RBF kernel. |
| `Nu` | Gets the nu parameter (upper bound on outlier fraction). |
| `NumSupportVectors` | Gets the number of support vectors after fitting. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

