---
title: "BIRCHOptions<T>"
description: "Configuration options for BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).

## For Beginners

BIRCH is like creating a summary tree of your data.

Imagine organizing a library:

1. First, create sections (CF-tree nodes)
2. Group similar books together (clustering features)
3. Each node summarizes: count, sum, sum of squares
4. Finally, cluster the summaries

Benefits:

- Handles very large datasets efficiently
- Incremental: can add data without rebuilding
- Single pass through data (mostly)
- Memory efficient with controllable tree size

Key parameters:

- Threshold: How similar points must be to join a cluster
- BranchingFactor: Max children per node (controls tree width)

## How It Works

BIRCH incrementally builds a CF (Clustering Feature) tree to summarize the data,
then applies a clustering algorithm to the leaf entries. It's designed for
very large datasets that don't fit in memory.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BIRCHOptions` | Initializes BIRCHOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BranchingFactor` | Gets or sets the maximum branching factor. |
| `ComputeLabels` | Gets or sets whether to compute cluster labels. |
| `Copy` | Gets or sets whether to copy the input data. |
| `DistanceMetric` | Gets or sets the distance metric. |
| `NumClusters` | Gets or sets the target number of clusters. |
| `Threshold` | Gets or sets the threshold for cluster radius. |

