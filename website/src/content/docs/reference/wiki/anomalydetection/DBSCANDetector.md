---
title: "DBSCANDetector<T>"
description: "Detects anomalies using DBSCAN (Density-Based Spatial Clustering of Applications with Noise)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.ClusterBased`

Detects anomalies using DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

## For Beginners

DBSCAN groups together points that are closely packed and
marks points in low-density regions as outliers (noise).

## How It Works

The algorithm works by:

1. For each point, find all neighbors within epsilon distance
2. If a point has at least minPts neighbors, it's a core point
3. Expand clusters from core points
4. Points not belonging to any cluster are noise (anomalies)

**When to use:**

- When anomalies are in low-density regions
- When clusters have arbitrary shapes
- No need to specify number of clusters in advance

**Industry Standard Defaults:**

- Epsilon: estimated from data (k-distance method)
- MinPts: 2 * dimensions
- Contamination: 0.1 (10%)

Reference: Ester, M., et al. (1996). "A Density-Based Algorithm for Discovering Clusters." KDD.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DBSCANDetector(Nullable<Double>,Nullable<Int32>,Double,Int32)` | Creates a new DBSCAN anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Epsilon` | Gets the epsilon (neighborhood radius) parameter. |
| `MinPts` | Gets the minimum points parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

