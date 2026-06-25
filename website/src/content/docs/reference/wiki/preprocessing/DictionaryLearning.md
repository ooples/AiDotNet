---
title: "DictionaryLearning<T>"
description: "Dictionary Learning for sparse representation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DimensionalityReduction`

Dictionary Learning for sparse representation.

## For Beginners

Dictionary Learning is like building a "parts library":

- Dictionary: Collection of basic building blocks (atoms)
- Sparse codes: Which parts to use and how much (mostly zeros)
- Goal: Represent each sample using few dictionary atoms
- Used for: Image denoising, compression, feature extraction

## How It Works

Dictionary Learning finds a dictionary D and sparse codes A such that
X ≈ D × A. The dictionary atoms (columns of D) form a basis that allows
sparse representation of the data.

Unlike PCA which enforces orthogonality, dictionary learning allows
overcomplete dictionaries (more atoms than dimensions) and enforces
sparsity on the codes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DictionaryLearning(Nullable<Int32>,Double,Int32,Double,DictionaryFitAlgorithm,SparseCodingAlgorithm,Nullable<Int32>,Int32[])` | Creates a new instance of `DictionaryLearning`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the sparsity penalty. |
| `Components` | Gets the dictionary atoms (each row is an atom). |
| `FitAlgorithm` | Gets the dictionary fitting algorithm. |
| `Mean` | Gets the mean of each feature. |
| `NComponents` | Gets the number of dictionary atoms. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `TransformAlgorithm` | Gets the sparse coding algorithm for transform. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Fits the dictionary using alternating minimization. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Transforms sparse codes back to original space. |
| `TransformCore(Matrix<>)` | Transforms data by computing sparse codes. |

