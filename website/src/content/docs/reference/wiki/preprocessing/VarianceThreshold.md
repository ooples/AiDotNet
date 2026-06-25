---
title: "VarianceThreshold<T>"
description: "Feature selector that removes features with variance below a threshold."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Feature selector that removes features with variance below a threshold.

## For Beginners

If a feature has the same value (or nearly the same value)
for all samples, it won't help your model distinguish between different outcomes.
This transformer automatically removes such features:

- Constant features (all same value) have variance = 0
- Near-constant features have very low variance

Example: A "Country" column that is "USA" for 99.9% of rows provides little information.

## How It Works

VarianceThreshold removes all features whose variance doesn't meet a minimum threshold.
Features with low variance are often not informative because they don't vary enough
across samples to be useful for prediction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VarianceThreshold(Double,Int32[])` | Creates a new instance of `VarianceThreshold`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SelectedFeatures` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `Threshold` | Gets the variance threshold. |
| `Variances` | Gets the computed variances for each feature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the variance of each feature. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets a boolean mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Removes features with variance below the threshold. |

