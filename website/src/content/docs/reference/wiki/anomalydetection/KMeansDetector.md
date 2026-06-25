---
title: "KMeansDetector<T>"
description: "Detects anomalies using K-Means clustering distance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AnomalyDetection.ClusterBased`

Detects anomalies using K-Means clustering distance.

## For Beginners

K-Means clustering groups data into k clusters based on distance
to cluster centers (centroids). Points far from their nearest centroid are considered
anomalies because they don't fit well into any cluster.

## How It Works

The algorithm works by:

1. Partition data into k clusters using K-Means
2. For each point, find distance to nearest centroid
3. Points with large distances are anomalies

**When to use:**

- When data naturally forms spherical clusters
- When you have a rough idea of the number of clusters
- Fast and scalable to large datasets

**Industry Standard Defaults:**

- K (clusters): 8 (or estimated from data)
- Max iterations: 100
- Contamination: 0.1 (10%)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KMeansDetector(Int32,Int32,Double,Int32)` | Creates a new K-Means anomaly detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `K` | Gets the number of clusters. |
| `MaxIterations` | Gets the maximum number of iterations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>)` |  |
| `ScoreAnomalies(Matrix<>)` |  |

