---
title: "NCA<T>"
description: "Neighborhood Components Analysis (NCA) for supervised dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Neighborhood Components Analysis (NCA) for supervised dimensionality reduction.

## For Beginners

NCA learns a space where k-NN works well:

- Points with same label are pulled closer together
- Points with different labels are pushed apart
- The learned transformation is linear (a matrix)
- Can be used for feature extraction before classification

Use cases:

- Preprocessing for k-NN classifiers
- Metric learning for distance-based methods
- Visualization with class structure preserved
- Feature extraction for classification tasks

## How It Works

NCA is a supervised dimensionality reduction algorithm that learns a linear transformation
to maximize the expected leave-one-out classification accuracy in the transformed space.
It uses stochastic neighbor assignment for soft nearest-neighbor classification.

The algorithm:

1. Define soft neighbor probabilities using softmax over distances
2. Compute expected leave-one-out classification accuracy
3. Optimize transformation matrix using gradient descent
4. Project data using learned transformation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NCA(Int32,Int32,Double,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `NCA`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NComponents` | Gets the number of components (dimensions). |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `TransformationMatrix` | Gets the learned transformation matrix. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Int32[])` | Fits NCA using the provided data and labels. |
| `FitCore(Matrix<>)` | Fits NCA (without labels - uses unsupervised initialization). |
| `FitTransformSupervised(Matrix<>,Int32[])` | Fits NCA using the provided data and labels, then transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms data using the learned transformation matrix. |

