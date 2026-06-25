---
title: "DunnIndex<T>"
description: "Computes the Dunn Index for cluster validity assessment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Computes the Dunn Index for cluster validity assessment.

## For Beginners

The Dunn Index asks two questions:

1. How far apart are different clusters? (larger = better)
2. How spread out is each cluster? (smaller = better)

A good clustering has:

- Large gaps between clusters
- Tight, compact clusters

The Dunn Index is the ratio: (smallest gap) / (largest spread)

- Higher values = better clustering
- Maximum when clusters are tight and well-separated

Limitations:

- Sensitive to outliers (a single far point increases diameter)
- Computationally expensive for large datasets (O(n²))

## How It Works

The Dunn Index is the ratio of the minimum inter-cluster distance to the
maximum intra-cluster distance. Higher values indicate better clustering.

Formula: D = min(d(C_i, C_j)) / max(diam(C_k))
where:

- d(C_i, C_j) = minimum distance between points in different clusters
- diam(C_k) = maximum distance between points within a cluster

## Properties

| Property | Summary |
|:-----|:--------|
| `HigherIsBetter` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Matrix<>,Vector<>)` |  |

