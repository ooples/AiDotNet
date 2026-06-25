---
title: "CUREOptions<T>"
description: "Configuration options for CURE clustering algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for CURE clustering algorithm.

## For Beginners

CURE is designed to find non-spherical clusters:

- Uses multiple "representative" points per cluster instead of just a center
- These points are pulled slightly toward the center (shrinking)
- This helps find oddly-shaped clusters (like bananas or spirals)

It's like describing a cluster by several key locations within it,
rather than just one center point.

## How It Works

CURE (Clustering Using REpresentatives) is a hierarchical clustering algorithm
that represents each cluster by a set of well-scattered representative points,
which are then shrunk toward the cluster center.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumClusters` | Gets or sets the number of clusters to find. |
| `NumPartitions` | Gets or sets the number of partitions for parallel processing. |
| `NumRepresentatives` | Gets or sets the number of representative points per cluster. |
| `SampleFraction` | Gets or sets the sample fraction for large datasets. |
| `ShrinkFactor` | Gets or sets the shrink factor for representative points. |
| `UsePartitioning` | Gets or sets whether to use random partitioning. |

