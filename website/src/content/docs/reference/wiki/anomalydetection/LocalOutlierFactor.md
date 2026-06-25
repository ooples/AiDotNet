---
title: "LocalOutlierFactor<T>"
description: "Implements the Local Outlier Factor (LOF) algorithm for density-based anomaly detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.DistanceBased`

Implements the Local Outlier Factor (LOF) algorithm for density-based anomaly detection.

## For Beginners

LOF detects anomalies by comparing the local density of a point
to the local densities of its neighbors. Points in low-density regions (relative to
their neighbors) are considered anomalies.

## How It Works

The algorithm works by:

1. Finding the k-nearest neighbors for each point
2. Computing the local reachability density (LRD) for each point
3. Comparing each point's LRD to its neighbors' LRDs to get the LOF score
4. Points with LOF score significantly greater than 1 are anomalies

**When to use:** LOF is particularly effective for:

- Detecting local anomalies (points that are anomalies relative to their neighborhood)
- Data with varying densities across different regions
- When you want interpretable density-based detection

**Industry Standard Defaults:**

- Number of neighbors (k): 20 (good balance between local and global detection)
- Contamination: 0.1 (10%)

Reference: Breunig, M. M., Kriegel, H. P., Ng, R. T., and Sander, J. (2000).
"LOF: Identifying Density-Based Local Outliers." ACM SIGMOD Record.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LocalOutlierFactor(Int32,Double,Int32)` | Creates a new Local Outlier Factor detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumNeighbors` | Gets the number of neighbors used for LOF computation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

