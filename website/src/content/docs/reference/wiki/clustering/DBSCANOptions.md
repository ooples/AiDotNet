---
title: "DBSCANOptions<T>"
description: "Configuration options for DBSCAN clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for DBSCAN clustering.

## For Beginners

DBSCAN works by finding dense regions in your data.

Key concepts:

- Epsilon (eps): The maximum distance between two points to be neighbors
- MinPoints: Minimum neighbors needed to form a dense region
- Core point: Has at least MinPoints neighbors within epsilon
- Border point: Near a core point but not enough neighbors to be core
- Noise: Points that don't belong to any cluster

Advantages over K-Means:

- No need to specify number of clusters
- Can find arbitrarily shaped clusters
- Robust to outliers (marks them as noise)
- Doesn't assume spherical clusters

Choosing parameters:

- eps: Use a k-distance graph (elbow method)
- MinPoints: 2 × dimensions is a good starting point

## How It Works

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters
based on density. Unlike K-Means, it doesn't require specifying the number of clusters
and can discover clusters of arbitrary shape.

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Gets or sets the algorithm for computing core sample neighborhoods. |
| `AutoEpsilon` | When true, automatically estimates epsilon from the data using the k-distance method. |
| `DistanceMetric` | Gets or sets the metric for distance calculations. |
| `Epsilon` | Gets or sets the epsilon radius for neighborhood queries. |
| `LeafSize` | Gets or sets the leaf size for BallTree or KDTree. |
| `MinPoints` | Gets or sets the minimum number of points to form a dense region. |
| `NumJobs` | Gets or sets the number of parallel jobs (-1 for all cores). |
| `P` | Gets or sets the power parameter for Minkowski distance. |

