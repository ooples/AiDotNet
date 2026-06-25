---
title: "CLIQUEOptions<T>"
description: "Configuration options for CLIQUE subspace clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for CLIQUE subspace clustering.

## For Beginners

CLIQUE finds clusters that may only exist in some dimensions:

- Divides each dimension into bins (like a grid)
- Finds dense cells (many points in a small region)
- Connects adjacent dense cells into clusters
- Works bottom-up: starts with 1D, extends to 2D, 3D, etc.

Key insight: Sometimes clusters only appear when looking at a few features,
not all of them. CLIQUE finds these hidden patterns.

## How It Works

CLIQUE (CLustering In QUEst) is a grid-based, density-based algorithm
for identifying clusters in subspaces of high-dimensional data.

## Properties

| Property | Summary |
|:-----|:--------|
| `DensityThreshold` | Gets or sets the density threshold as a fraction of total points. |
| `MaxSubspaceDimensions` | Gets or sets the maximum subspace dimensionality to explore. |
| `MinPoints` | Gets or sets the minimum number of points for a dense unit. |
| `NumIntervals` | Gets or sets the number of intervals (bins) per dimension. |
| `UseAprioriPruning` | Gets or sets whether to prune using the Apriori principle. |

