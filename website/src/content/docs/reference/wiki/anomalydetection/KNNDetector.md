---
title: "KNNDetector<T>"
description: "Detects anomalies using K-Nearest Neighbors distance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.DistanceBased`

Detects anomalies using K-Nearest Neighbors distance.

## For Beginners

KNN anomaly detection identifies outliers based on their distance
to their k-nearest neighbors. Points that are far from their nearest neighbors are
considered anomalies.

## How It Works

The algorithm works by:

1. For each point, find the k nearest neighbors
2. Calculate the average distance to these neighbors
3. Points with large average distances are anomalies

**When to use:**

- When anomalies are expected to be isolated from normal clusters
- Works well with low-to-medium dimensional data
- No assumption about data distribution

**Industry Standard Defaults:**

- K (neighbors): 5 (typical range 3-20)
- Contamination: 0.1 (10%)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KNNDetector(Int32,Double,Int32)` | Creates a new KNN anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `K` | Gets the number of neighbors used for detection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

