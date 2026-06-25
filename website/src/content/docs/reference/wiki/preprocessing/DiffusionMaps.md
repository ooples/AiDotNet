---
title: "DiffusionMaps<T>"
description: "Diffusion Maps for nonlinear dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Diffusion Maps for nonlinear dimensionality reduction.

## For Beginners

Diffusion Maps captures the underlying geometry by:

- Simulating how information "diffuses" through the data
- Points connected by many short paths are close in diffusion distance
- Robust to noise compared to geodesic distances

Use cases:

- Discovering underlying manifold structure
- Robust to noise in the data
- When you want distances to reflect connectivity, not just proximity

## How It Works

Diffusion Maps embeds data into a low-dimensional space where Euclidean distances
approximate diffusion distances on the data manifold. It simulates a random walk
on the data graph and uses the eigenvectors of the diffusion operator.

The algorithm:

1. Constructs a kernel matrix (typically Gaussian)
2. Normalizes to create a Markov transition matrix
3. Computes eigenvectors of the diffusion operator
4. Scales eigenvectors by eigenvalue powers for embedding

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionMaps(Int32,Double,Int32,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `DiffusionMaps`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiffusionTime` | Gets the diffusion time parameter. |
| `Eigenvalues` | Gets the eigenvalues. |
| `Embedding` | Gets the embedding result. |
| `NComponents` | Gets the number of components (dimensions). |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits Diffusion Maps and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

