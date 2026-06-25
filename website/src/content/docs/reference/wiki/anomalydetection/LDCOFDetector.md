---
title: "LDCOFDetector<T>"
description: "Detects anomalies using LDCOF (Local Density Cluster-Based Outlier Factor)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.DistanceBased`

Detects anomalies using LDCOF (Local Density Cluster-Based Outlier Factor).

## For Beginners

LDCOF combines clustering with density-based outlier detection.
It first clusters the data, then computes outlier scores based on how a point's density
compares to its cluster's density. Points in sparse regions of dense clusters are flagged.

## How It Works

The algorithm works by:

1. Cluster the data using k-means
2. Compute local density for each point
3. Compare point density to cluster average density
4. Large deviations indicate outliers

**When to use:**

- When data has natural cluster structure
- For detecting local outliers within clusters
- When global methods miss cluster-specific anomalies

**Industry Standard Defaults:**

- Number of clusters: 8
- Number of neighbors (k): 10
- Contamination: 0.1 (10%)

Reference: Amer, M., Goldstein, M. (2012). "Nearest-Neighbor and Clustering based
Anomaly Detection Algorithms for RapidMiner." Workshop on Open Source Data Mining Software.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LDCOFDetector(Int32,Int32,Double,Int32)` | Creates a new LDCOF anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClusters` | Gets the number of clusters. |
| `NumNeighbors` | Gets the number of neighbors for density estimation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

