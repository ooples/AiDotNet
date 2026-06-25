---
title: "SelectPercentile<T>"
description: "Selects features according to a percentile of the highest scores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Selects features according to a percentile of the highest scores.

## For Beginners

Instead of specifying an exact number of features:

- SelectPercentile(50) keeps the top 50% of features
- The actual number depends on your original feature count
- Useful when you want a proportion, not an absolute count

## How It Works

SelectPercentile selects the top percentile of features based on a scoring
function. For example, selecting the top 10% of features ranked by F-score.

This is similar to SelectKBest but uses a relative threshold (percentile)
instead of an absolute number of features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelectPercentile(Double,SelectKBestScoreFunc,Int32[])` | Creates a new instance of `SelectPercentile`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PValues` | Gets the p-values for each feature. |
| `Percentile` | Gets the percentile of features to select (0-100). |
| `Scores` | Gets the scores for each feature. |
| `ScoringFunction` | Gets the scoring function used. |
| `SelectedIndices` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the selector by computing feature scores. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets the support mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting top percentile features. |

