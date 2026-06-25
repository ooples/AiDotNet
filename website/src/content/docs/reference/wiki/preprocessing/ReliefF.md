---
title: "ReliefF<T>"
description: "ReliefF feature selection algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Distance`

ReliefF feature selection algorithm.

## For Beginners

ReliefF scores each feature by asking:
"Does this feature help distinguish different classes?"

If instances from the same class have similar values but instances from
different classes have different values, the feature gets a high score.

Unlike correlation-based methods, ReliefF can detect feature interactions
and works well with non-linear relationships.

## How It Works

ReliefF is a feature weighting algorithm that estimates feature quality based on
how well features distinguish between instances that are near each other.

For each randomly sampled instance, ReliefF:

1. Finds k nearest neighbors from the same class (hits)
2. Finds k nearest neighbors from each different class (misses)
3. Updates feature weights based on distance differences

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReliefF(Int32,Int32,Int32,Nullable<Int32>,Int32[])` | Creates a new instance of `ReliefF`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureWeights` | Gets the computed feature weights. |
| `NFeaturesToSelect` | Gets the number of features to select. |
| `NNeighbors` | Gets the number of nearest neighbors used. |
| `SelectedIndices` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits ReliefF by computing feature weights. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets a boolean mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting ReliefF-ranked features. |

