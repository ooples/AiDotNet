---
title: "TSNE<T>"
description: "t-Distributed Stochastic Neighbor Embedding for visualization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

t-Distributed Stochastic Neighbor Embedding for visualization.

## For Beginners

t-SNE creates beautiful 2D/3D visualizations:

- Points that are similar stay close together
- Points that are different move apart
- Great for exploring clusters in your data
- Warning: Not for preserving global distances, just local neighborhoods
- Warning: Results can vary with different random seeds

## How It Works

t-SNE is a nonlinear dimensionality reduction technique well-suited for
visualizing high-dimensional data in 2D or 3D space. It preserves local
structure by keeping similar points close together.

The algorithm converts similarities between data points to joint probabilities
and tries to minimize the divergence between probability distributions in
high and low dimensional spaces.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TSNE(Int32,Double,Double,Int32,Double,TSNEMetric,TSNEInitialization,Nullable<Int32>,Int32[])` | Creates a new instance of `TSNE`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Embedding` | Gets the embedding result. |
| `LearningRate` | Gets the learning rate. |
| `Metric` | Gets the distance metric. |
| `NComponents` | Gets the number of components (dimensions). |
| `Perplexity` | Gets the perplexity parameter. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits t-SNE and computes the embedding. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Returns the embedding computed during Fit. |

