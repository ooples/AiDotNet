---
title: "MeanShiftOptions<T>"
description: "Configuration options for Mean Shift clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for Mean Shift clustering.

## For Beginners

Mean Shift finds the "peaks" in your data.

Imagine your data as a landscape with hills:

- Each point starts at its location
- Points "roll uphill" toward the nearest peak
- Points that end up at the same peak form a cluster

Key features:

- Automatically determines number of clusters
- Works well for finding dense regions
- Good for image segmentation

Main parameter: bandwidth (how wide the "hills" are)

- Small bandwidth: Many small clusters
- Large bandwidth: Few large clusters

## How It Works

Mean Shift is a non-parametric clustering algorithm that doesn't require
specifying the number of clusters. It finds clusters by iteratively shifting
points toward the mode (densest area) of the local density.

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Gets or sets the neighbor finding algorithm. |
| `Bandwidth` | Gets or sets the bandwidth parameter. |
| `BandwidthQuantile` | Gets or sets the quantile for automatic bandwidth estimation. |
| `BinSeeding` | Gets or sets whether to bin the seeds to speed up computation. |
| `ClusterAll` | Gets or sets whether to cluster all points or just the seeds. |
| `ClusterMergeThreshold` | Gets or sets the minimum distance between cluster centers. |
| `DistanceMetric` | Gets or sets the distance metric. |
| `LeafSize` | Gets or sets the leaf size for tree algorithms. |
| `MinBinFrequency` | Gets or sets the minimum number of points to consider a cluster. |
| `NumJobs` | Gets or sets the number of parallel jobs. |

