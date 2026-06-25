---
title: "SelectFdr<T>"
description: "Selects features based on a false discovery rate test."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Selects features based on a false discovery rate test.

## For Beginners

FDR correction is more powerful than FPR:

- FPR: "At most 5% of all 1000 features are false positives"
- FDR: "At most 5% of the 50 features I selected are false positives"
- FDR allows more features to be selected while controlling errors

## How It Works

SelectFdr applies the Benjamini-Hochberg procedure to control the expected
proportion of false discoveries (incorrectly selected features) among the
selected features.

Unlike SelectFpr which controls false positives among ALL features,
SelectFdr controls false discoveries among SELECTED features only.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelectFdr(Double,SelectKBestScoreFunc,Int32[])` | Creates a new instance of `SelectFdr`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdjustedPValues` | Gets the FDR-adjusted p-values. |
| `Alpha` | Gets the significance level (alpha) for FDR control. |
| `PValues` | Gets the original p-values for each feature. |
| `Scores` | Gets the scores for each feature. |
| `ScoringFunction` | Gets the scoring function used. |
| `SelectedIndices` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the selector by computing FDR-adjusted p-values. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets the support mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting features passing FDR threshold. |

