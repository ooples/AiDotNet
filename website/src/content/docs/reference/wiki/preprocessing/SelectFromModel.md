---
title: "SelectFromModel<T>"
description: "Selects features based on importance weights from an external model or scorer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Selects features based on importance weights from an external model or scorer.

## For Beginners

This works with any model that produces feature importances:

- Random forests give importance based on how much each feature reduces error
- Linear models give coefficients showing feature influence
- Features below the threshold are removed

## How It Works

SelectFromModel selects features based on importance scores, typically from a fitted model.
Features with importance above a threshold are kept.

The threshold can be specified as:

- An absolute value
- "mean" - the mean of feature importances
- "median" - the median of feature importances

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelectFromModel(Double[],SelectFromModelThreshold,Double,Nullable<Int32>,Int32[])` | Creates a new instance with precomputed feature importances. |
| `SelectFromModel(Func<Matrix<>,Vector<>,Double[]>,SelectFromModelThreshold,Double,Nullable<Int32>,Int32[])` | Creates a new instance with a function to compute feature importances. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureImportances` | Gets the feature importances used for selection. |
| `SelectedIndices` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `Threshold` | Gets the computed threshold value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the selector by computing feature importances. |
| `FitCore(Matrix<>)` | Fits the selector using precomputed importances only. |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets the support mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting important features. |

