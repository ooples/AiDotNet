---
title: "UMAP<T>"
description: "Uniform Manifold Approximation and Projection for dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Uniform Manifold Approximation and Projection for dimensionality reduction.

## For Beginners

UMAP creates visualizations similar to t-SNE but:

- It's faster, especially for large datasets
- Distances between clusters are more meaningful
- You can transform new data points without refitting
- Great for both visualization AND as a preprocessing step for ML

Example use cases:

- Visualizing high-dimensional data (gene expression, embeddings)
- Preprocessing features for classification
- Clustering analysis
- Anomaly detection

## How It Works

UMAP is a nonlinear dimensionality reduction technique that constructs a high-dimensional
graph representation and optimizes a low-dimensional graph to be as structurally similar
as possible. It is based on Riemannian geometry and algebraic topology.

Key advantages over t-SNE:

- Much faster (scales better to large datasets)
- Preserves more global structure
- Supports out-of-sample transformation
- More deterministic results

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UMAP(Int32,Int32,Double,Double,UMAPMetric,Int32,Double,Double,Double,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `UMAP`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Embedding` | Gets the embedding result. |
| `Metric` | Gets the distance metric. |
| `MinDist` | Gets the minimum distance parameter. |
| `NComponents` | Gets the number of components (dimensions). |
| `NNeighbors` | Gets the number of neighbors. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistance(Double[],Double[])` | Computes distance between two points using the configured metric. |
| `FitCore(Matrix<>)` | Fits UMAP and computes the embedding. |
| `GetEmbedding` | Gets the embedding computed during Fit for the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms data using the fitted UMAP embedding. |

