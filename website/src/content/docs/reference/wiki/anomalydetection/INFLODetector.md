---
title: "INFLODetector<T>"
description: "Detects anomalies using Influenced Outlierness (INFLO)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.DistanceBased`

Detects anomalies using Influenced Outlierness (INFLO).

## For Beginners

INFLO combines the concepts of k-nearest neighbors and reverse
nearest neighbors. It considers not only which points are close to a given point,
but also which points consider the given point as their neighbor.

## How It Works

The algorithm works by:

1. Find k-nearest neighbors (kNN) for each point
2. Find reverse k-nearest neighbors (RkNN) - points that have this point as a neighbor
3. Compute influence space = kNN union RkNN
4. Compare point's density to its influence space's density

**When to use:**

- When boundary between clusters have outliers
- When LOF fails at cluster boundaries
- Similar scenarios to LOF but with better boundary handling

**Industry Standard Defaults:**

- K (neighbors): 10
- Contamination: 0.1 (10%)

Reference: Jin, W., et al. (2006). "Mining Top-n Local Outliers in Large Databases." KDD.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `INFLODetector(Int32,Double,Int32)` | Creates a new INFLO anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `K` | Gets the number of neighbors used for detection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

