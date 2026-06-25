---
title: "FastICA<T>"
description: "Fast Independent Component Analysis (FastICA) for blind source separation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Fast Independent Component Analysis (FastICA) for blind source separation.

## For Beginners

Imagine you have recordings from multiple microphones
in a room with multiple speakers. ICA can separate the individual speakers:

- It finds components that are statistically independent
- Unlike PCA which finds uncorrelated components, ICA finds truly independent ones
- Works best when source signals are non-Gaussian (most real-world signals are)

## How It Works

FastICA separates a multivariate signal into additive, independent non-Gaussian
components. It is commonly used for blind source separation (e.g., separating
mixed audio signals) and feature extraction.

The algorithm:

1. Centers and whitens the data
2. Uses fixed-point iteration to find independent components
3. Each component is found by maximizing non-Gaussianity

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastICA(Int32,ICAAlgorithm,ICAFunction,Int32,Double,Boolean,Int32,Int32[])` | Creates a new instance of `FastICA`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Gets the algorithm type (parallel or deflation). |
| `Components` | Gets the unmixing matrix (components). |
| `Fun` | Gets the non-linearity function used. |
| `Mixing` | Gets the mixing matrix. |
| `NComponents` | Gets the number of independent components. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits FastICA to extract independent components. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Transforms independent components back to original space. |
| `TransformCore(Matrix<>)` | Transforms the data to independent components. |

