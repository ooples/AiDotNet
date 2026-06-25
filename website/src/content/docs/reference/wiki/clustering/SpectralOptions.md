---
title: "SpectralOptions<T>"
description: "Configuration options for Spectral Clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for Spectral Clustering.

## For Beginners

Spectral clustering finds clusters by analyzing connections.

Think of your data as a graph where:

- Each point is a node
- Similar points are connected by edges
- Clusters are groups of densely connected nodes

The algorithm:

1. Build a similarity graph (like a social network)
2. Find the "natural cuts" in this graph
3. These cuts define your clusters

When to use:

- Clusters have unusual shapes (crescents, spirals)
- You can define a good similarity measure
- Data has clear connectivity patterns

## How It Works

Spectral clustering uses the eigenvalues of a similarity matrix to perform
dimensionality reduction before clustering. It can find non-convex clusters
that K-Means cannot.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralOptions` | Initializes a new instance of SpectralOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Affinity` | Gets or sets the affinity/similarity method. |
| `AssignLabels` | Gets or sets the assignment strategy after spectral embedding. |
| `DistanceMetric` | Gets or sets the distance metric for building affinity matrix. |
| `EigenSolver` | Gets or sets the eigenvalue solver method. |
| `Gamma` | Gets or sets the gamma parameter for RBF kernel. |
| `Normalization` | Gets or sets the type of Laplacian normalization. |
| `NumClusters` | Gets or sets the number of clusters. |
| `NumNeighbors` | Gets or sets the number of neighbors for nearest neighbors affinity. |

