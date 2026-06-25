---
title: "HierarchicalOptions<T>"
description: "Configuration options for Hierarchical (Agglomerative) clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for Hierarchical (Agglomerative) clustering.

## For Beginners

Hierarchical clustering creates a family tree of data.

How it works:

1. Start with each point as its own cluster
2. Find the two closest clusters
3. Merge them into one
4. Repeat until desired number of clusters

The result is a dendrogram (tree diagram) showing how clusters merge.
You can cut the tree at different levels to get different numbers of clusters.

Linkage methods determine "closest":

- Single: Nearest points in clusters (chain-like clusters)
- Complete: Farthest points in clusters (compact clusters)
- Average: Average of all pairwise distances
- Ward: Minimizes within-cluster variance (most popular)

## How It Works

Hierarchical clustering builds a tree (dendrogram) of clusters by successively
merging or splitting clusters based on distance. Agglomerative clustering
starts with each point as its own cluster and merges them bottom-up.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComputeDistances` | Gets or sets whether to compute distances between clusters. |
| `ComputeFullTree` | Gets or sets whether to compute the full dendrogram. |
| `Connectivity` | Gets or sets the connectivity constraint matrix. |
| `DistanceMetric` | Gets or sets the distance metric to use. |
| `DistanceThreshold` | Gets or sets the distance threshold for cluster formation. |
| `Linkage` | Gets or sets the linkage criterion for merging clusters. |
| `NumClusters` | Gets or sets the number of clusters to find. |

