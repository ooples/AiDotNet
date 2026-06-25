---
title: "SelectKBest<T>"
description: "Selects the K highest-scoring features according to a scoring function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Selects the K highest-scoring features according to a scoring function.

## For Beginners

Not all features are equally useful for prediction.
SelectKBest helps you:

- Reduce the number of features to improve model speed
- Remove noisy features that might hurt model accuracy
- Find the most informative features for understanding your problem

Example: From 100 features, select the 10 most related to your target.

## How It Works

SelectKBest computes a score for each feature based on the relationship between
the feature and the target variable, then selects the top K features with the
highest scores.

Built-in scoring functions include:

- F-score for regression (linear relationship)
- Mutual information (any relationship type)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelectKBest(Int32,SelectKBestScoreFunc,Int32[])` | Creates a new instance of `SelectKBest`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `K` | Gets the number of features to select. |
| `PValues` | Gets the p-values for each feature (if applicable). |
| `ScoreFunc` | Gets the scoring function used. |
| `Scores` | Gets the computed scores for each feature. |
| `SelectedFeatures` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the selector by computing feature scores based on the target. |
| `FitCore(Matrix<>)` | Fits the selector using variance-based scoring when no target is provided. |
| `FitTransform(Matrix<>,Vector<>)` | Fits the selector and transforms the data in one step. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets a boolean mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Selects the top K features from the data. |

