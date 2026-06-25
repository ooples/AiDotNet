---
title: "PHATE<T>"
description: "PHATE: Potential of Heat-diffusion for Affinity-based Transition Embedding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

PHATE: Potential of Heat-diffusion for Affinity-based Transition Embedding.

## For Beginners

PHATE excels at revealing data trajectories:

- Captures smooth progression paths in data (differentiation, time courses)
- Preserves both local clusters and global connectivity
- The potential distance emphasizes transitions between states
- Particularly effective for biological data

Use cases:

- Single-cell RNA sequencing visualization
- Developmental trajectories and cell differentiation
- Time-series data with smooth transitions
- Any data with underlying continuous processes

## How It Works

PHATE is a dimensionality reduction method designed for visualizing high-dimensional
biological data, particularly single-cell data. It captures both local and global
structure by using diffusion-based distances and a special potential distance metric.

The algorithm:

1. Compute local affinities using adaptive bandwidth Gaussian kernel
2. Construct diffusion operator from affinities
3. Diffuse for t steps to capture multi-scale structure
4. Compute potential distance using log transform
5. Embed using MDS on potential distances

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PHATE(Int32,Int32,Int32,Double,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `PHATE`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiffusionTime` | Gets the diffusion time parameter. |
| `Embedding` | Gets the embedding result. |
| `NComponents` | Gets the number of components (dimensions). |
| `NNeighbors` | Gets the number of neighbors. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits PHATE and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

