---
title: "CBLOFDetector<T>"
description: "Detects anomalies using Cluster-Based Local Outlier Factor (CBLOF)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.ClusterBased`

Detects anomalies using Cluster-Based Local Outlier Factor (CBLOF).

## For Beginners

CBLOF combines clustering with local outlier detection.
It first clusters the data, then scores each point based on its distance to
its cluster center, weighted by the cluster size.

## How It Works

The algorithm works by:

1. Cluster data into large and small clusters
2. For large cluster points: score = cluster_size * distance_to_center
3. For small cluster points: score = cluster_size * distance_to_nearest_large_cluster

**When to use:**

- When anomalies are expected in small clusters or far from large clusters
- Faster than LOF for large datasets
- When cluster structure is meaningful

**Industry Standard Defaults:**

- K (clusters): 8
- Alpha (large cluster threshold): 0.9 (90% of data)
- Beta (size threshold): 5 points
- Contamination: 0.1 (10%)

Reference: He, Z., et al. (2003). "Discovering Cluster-Based Local Outliers." Pattern Recognition Letters.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CBLOFDetector(Int32,Double,Int32,Double,Int32)` | Creates a new CBLOF anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the alpha parameter (proportion for large clusters). |
| `Beta` | Gets the beta parameter (minimum size for large clusters). |
| `NClusters` | Gets the number of clusters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

