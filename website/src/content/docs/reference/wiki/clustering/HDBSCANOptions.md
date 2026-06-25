---
title: "HDBSCANOptions<T>"
description: "Configuration options for HDBSCAN (Hierarchical DBSCAN)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for HDBSCAN (Hierarchical DBSCAN).

## For Beginners

HDBSCAN finds clusters at all density levels.

Problems with DBSCAN:

- Need to pick epsilon (cluster radius) - hard to choose!
- One epsilon can't handle both dense and sparse areas

HDBSCAN solution:

- Try ALL possible epsilon values automatically
- Build a tree of how clusters merge as density changes
- Pick the most "stable" clusters from this tree

Benefits:

- No epsilon parameter needed
- Finds clusters of varying densities
- Only need minClusterSize (intuitive parameter)
- Robust noise detection

## How It Works

HDBSCAN extends DBSCAN by constructing a hierarchy of clusters at different
density levels, then extracting flat clusters using a stability-based method.
It handles clusters of varying densities better than DBSCAN.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllowSingleCluster` | Gets or sets whether to allow single-cluster result. |
| `Alpha` | Gets or sets the alpha value for metric. |
| `ClusterSelection` | Gets or sets the cluster selection method. |
| `ClusterSelectionEpsilon` | Gets or sets the cluster selection epsilon. |
| `DistanceMetric` | Gets or sets the distance metric. |
| `MinClusterSize` | Gets or sets the minimum cluster size. |
| `MinSamples` | Gets or sets the minimum samples for core points. |

