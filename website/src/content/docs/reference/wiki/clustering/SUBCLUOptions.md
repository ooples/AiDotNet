---
title: "SUBCLUOptions<T>"
description: "Configuration options for SUBCLU subspace clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for SUBCLU subspace clustering.

## For Beginners

SUBCLU finds clusters hiding in feature subsets:

- Uses DBSCAN's concept of density-connected clusters
- Exploits the fact that if there's no cluster in 2D, there can't be one in 3D+
- This pruning makes it much faster than testing all possible subspaces

Great for datasets with many features where clusters only exist in some dimensions.

## How It Works

SUBCLU (SUBspace CLUstering) is a density-connected subspace clustering algorithm
that uses DBSCAN as its base clustering method and the monotonicity of density-connected
clusters to efficiently prune the subspace search.

## Properties

| Property | Summary |
|:-----|:--------|
| `Epsilon` | Gets or sets the epsilon radius for DBSCAN density computation. |
| `MaxSubspaceDimensions` | Gets or sets the maximum subspace dimensionality to explore. |
| `MinClusterSize` | Gets or sets the minimum cluster size to keep. |
| `MinPoints` | Gets or sets the minimum points required for a dense region. |

