---
title: "OCSVMDetector<T>"
description: "Detects anomalies using One-Class SVM with simplified SGD training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.DistanceBased`

Detects anomalies using One-Class SVM with simplified SGD training.

## For Beginners

One-Class SVM finds a boundary that encompasses the normal data.
It learns the shape of normal data in a high-dimensional kernel space and flags points
outside this region as anomalies.

## How It Works

The algorithm works by:

1. Map data to kernel space using RBF kernel
2. Find hyperplane that separates data from origin with maximum margin
3. Points on the negative side of the hyperplane are anomalies

**When to use:**

- When you have only normal examples
- High-dimensional data
- When you need a decision boundary

**Industry Standard Defaults:**

- Nu: 0.1 (roughly proportion of outliers)
- Gamma: auto (1/n_features)
- Kernel: RBF
- Contamination: 0.1 (10%)

Reference: Schölkopf, B., et al. (2001). "Estimating the Support of a High-Dimensional Distribution."

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OCSVMDetector(Double,Double,Int32,Double,Int32)` | Creates a new OCSVM anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Gamma` | Gets the gamma parameter. |
| `Nu` | Gets the nu parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

