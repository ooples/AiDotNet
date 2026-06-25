---
title: "Isomap<T>"
description: "Isomap (Isometric Mapping) for nonlinear dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Isomap (Isometric Mapping) for nonlinear dimensionality reduction.

## For Beginners

Isomap "unrolls" curved data:

- Imagine data lying on a Swiss roll (curved surface)
- Regular PCA can't flatten it properly
- Isomap finds distances along the surface, not through the air
- Result: Points that are nearby on the surface stay nearby

## How It Works

Isomap extends classical MDS by estimating geodesic distances along the data
manifold instead of using Euclidean distances. It builds a neighborhood graph
and computes shortest paths to estimate geodesic distances.

The algorithm:

1. Build a k-nearest neighbors graph
2. Compute shortest path distances (Floyd-Warshall or Dijkstra)
3. Apply MDS to the geodesic distance matrix

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Isomap(Int32,Int32,IsomapNeighborAlgorithm,IsomapPathAlgorithm,Int32[])` | Creates a new instance of `Isomap`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Embedding` | Gets the embedding result. |
| `GeodesicDistances` | Gets the geodesic distance matrix. |
| `NComponents` | Gets the number of components. |
| `NNeighbors` | Gets the number of neighbors for graph construction. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits Isomap by computing geodesic distances and embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

