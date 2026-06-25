---
title: "SelectFpr<T>"
description: "Selects features based on a false positive rate test."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Selects features based on a false positive rate test.

## For Beginners

This selector uses statistical significance:

- Computes a p-value for each feature (how likely it's just noise)
- Keeps features with p-value below alpha (threshold)
- Lower alpha = stricter selection = fewer false positives

## How It Works

SelectFpr selects features whose p-value is below a threshold (alpha).
This controls the expected percentage of false positives among all features.

For example, with alpha=0.05, we expect about 5% of the selected features
to be false positives (features that passed the test by chance).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelectFpr(Double,SelectKBestScoreFunc,Int32[])` | Creates a new instance of `SelectFpr`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the significance level (alpha). |
| `PValues` | Gets the p-values for each feature. |
| `Scores` | Gets the scores for each feature. |
| `ScoringFunction` | Gets the scoring function used. |
| `SelectedIndices` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the selector by computing p-values. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets the support mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting significant features. |

