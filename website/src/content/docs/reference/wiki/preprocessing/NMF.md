---
title: "NMF<T>"
description: "Non-negative Matrix Factorization for dimensionality reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Non-negative Matrix Factorization for dimensionality reduction.

## For Beginners

NMF learns "building blocks" from your data:

- For images: NMF might learn eyes, noses, mouths as separate components
- For text: NMF might learn topics as combinations of words
- Unlike PCA, components are always positive (additive, never subtractive)

Think of it as finding the parts that, when added together, recreate your data.

## How It Works

NMF factorizes a non-negative matrix V into two non-negative matrices W and H
such that V ≈ W × H. This is useful for data that is naturally non-negative
(e.g., images, text term frequencies, audio spectrograms).

Unlike PCA, NMF produces parts-based representations where each component
represents an additive combination of non-negative features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NMF(Int32,NMFInit,NMFSolver,Double,Double,Double,Int32,Double,Int32,Int32[])` | Creates a new instance of `NMF`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Components` | Gets the learned components (H matrix). |
| `NComponents` | Gets the number of components. |
| `NIterations` | Gets the number of iterations performed. |
| `ReconstructionError` | Gets the final reconstruction error. |
| `Solver` | Gets the solver type. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits NMF by learning the components matrix. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Reconstructs data from coefficient matrix. |
| `TransformCore(Matrix<>)` | Transforms data by finding the optimal W given learned H. |

