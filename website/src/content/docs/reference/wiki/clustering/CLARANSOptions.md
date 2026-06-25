---
title: "CLARANSOptions<T>"
description: "Configuration options for CLARANS (Clustering Large Applications based on Randomized Search)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for CLARANS (Clustering Large Applications based on Randomized Search).

## For Beginners

CLARANS randomly explores different cluster configurations.

Think of it like:

1. Start with random cluster representatives (medoids)
2. Try swapping representatives with random other points
3. Keep changes that improve clustering quality
4. Repeat until satisfied

Benefits over K-Means:

- Cluster centers are actual data points (medoids)
- More robust to outliers
- Works with any distance function

Trade-offs:

- Slower than K-Means for large datasets
- Randomized: may find different solutions each run

## How It Works

CLARANS is a medoid-based algorithm that uses randomized sampling to efficiently
search for good cluster medoids. It's more scalable than PAM while maintaining
the benefit of using actual data points as cluster centers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CLARANSOptions` | Initializes CLARANSOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistanceMetric` | Gets or sets the distance metric. |
| `MaxNeighbor` | Gets or sets the number of local search iterations. |
| `NumClusters` | Gets or sets the number of clusters. |
| `NumLocal` | Gets or sets the number of local minima to find. |

