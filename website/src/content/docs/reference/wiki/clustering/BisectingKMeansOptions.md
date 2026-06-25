---
title: "BisectingKMeansOptions<T>"
description: "Configuration options for Bisecting K-Means clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for Bisecting K-Means clustering.

## For Beginners

Bisecting K-Means works by:

1. Starting with all data in one big cluster
2. Splitting the largest cluster into two using K-Means
3. Repeating step 2 until you have k clusters

Advantages over regular K-Means:

- Often produces better clusters (more balanced)
- Less sensitive to initialization
- Builds a cluster hierarchy as a side effect

## How It Works

Bisecting K-Means is a divisive hierarchical clustering algorithm that
starts with all points in one cluster and recursively bisects clusters
until the desired number of clusters is reached.

## Properties

| Property | Summary |
|:-----|:--------|
| `BuildHierarchy` | Gets or sets whether to build a hierarchy tree during clustering. |
| `ClusterSelection` | Gets or sets the cluster selection method for bisection. |
| `MinClusterSizeForBisection` | Gets or sets the minimum cluster size that can be bisected. |
| `NumBisectionTrials` | Gets or sets the number of bisection trials at each split. |
| `NumClusters` | Gets or sets the number of clusters to find. |

