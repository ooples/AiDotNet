---
title: "GenericUnivariateSelect<T>"
description: "Generic univariate feature selector with configurable mode."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Generic univariate feature selector with configurable mode.

## For Beginners

This is a "swiss army knife" selector:

- k_best: Select exactly k features
- percentile: Select top X% of features
- fpr: Select based on false positive rate
- fdr: Select based on false discovery rate
- fwe: Select based on family-wise error rate

## How It Works

GenericUnivariateSelect provides a unified interface to all univariate
feature selection methods. You can choose between k_best, percentile,
fpr, fdr, and fwe modes.

This is useful when you want to experiment with different selection
strategies without changing your code structure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GenericUnivariateSelect(UnivariateSelectMode,Object,SelectKBestScoreFunc,Int32[])` | Creates a new instance of `GenericUnivariateSelect`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Mode` | Gets the selection mode. |
| `PValues` | Gets the p-values for each feature. |
| `Param` | Gets the mode parameter (k, percentile, or alpha depending on mode). |
| `Scores` | Gets the scores for each feature. |
| `ScoringFunction` | Gets the scoring function used. |
| `SelectedIndices` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the selector by computing scores and selecting features. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets the support mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting features. |

