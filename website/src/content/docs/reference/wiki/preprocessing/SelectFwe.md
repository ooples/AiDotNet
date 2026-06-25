---
title: "SelectFwe<T>"
description: "Selects features based on a family-wise error rate test."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Selects features based on a family-wise error rate test.

## For Beginners

FWER is the strictest correction:

- Controls the probability of ANY false positive
- Uses Bonferroni: alpha/number_of_features
- Very conservative: may miss true positives
- Best when false positives are costly

## How It Works

SelectFwe applies Bonferroni correction to control the probability of making
even one false positive among all selected features.

This is the most conservative multiple testing correction, dividing the
significance threshold by the number of tests.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelectFwe(Double,SelectKBestScoreFunc,Int32[])` | Creates a new instance of `SelectFwe`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdjustedPValues` | Gets the Bonferroni-adjusted p-values. |
| `Alpha` | Gets the family-wise significance level (alpha). |
| `PValues` | Gets the original p-values for each feature. |
| `Scores` | Gets the scores for each feature. |
| `ScoringFunction` | Gets the scoring function used. |
| `SelectedIndices` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the selector by computing Bonferroni-corrected p-values. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets the support mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting features passing FWER threshold. |

