---
title: "KMeansOptions<T>"
description: "Configuration options specific to KMeans clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options specific to KMeans clustering.

## For Beginners

KMeans works by:

1. Choosing k initial center points
2. Assigning each data point to its nearest center
3. Recalculating centers as the mean of assigned points
4. Repeating steps 2-3 until centers stabilize

The key settings are:

- Number of clusters (k): How many groups to find
- Initialization method: How to choose starting centers
- Number of runs: How many times to try different starting points

## How It Works

KMeans is a centroid-based clustering algorithm that partitions n observations
into k clusters where each observation belongs to the cluster with the nearest centroid.

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Gets or sets the algorithm variant to use. |
| `CopyX` | Gets or sets whether to copy input data for safety. |
| `InitMethod` | Gets or sets the initialization method for cluster centers. |
| `InitialCenters` | Gets or sets custom initial cluster centers. |
| `NumClusters` | Gets or sets the number of clusters to find. |
| `PrecomputeDistances` | Gets or sets whether to precompute distances. |

