---
title: "FisherScore<T>"
description: "Fisher Score feature selection for classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Distance`

Fisher Score feature selection for classification.

## For Beginners

Fisher Score answers the question:
"How well does this feature separate different classes?"

A high Fisher Score means:

- Classes have very different average values for this feature (high between-class variance)
- Within each class, values are similar (low within-class variance)

Example: If predicting "is_spam", a feature like "number of exclamation marks"
might have a high Fisher Score because spam emails have many (!!!!) while
normal emails have few.

## How It Works

Fisher Score measures the discriminative power of each feature by computing
the ratio of between-class variance to within-class variance.

For feature f, the Fisher Score is:

where μ_c is the mean of feature f in class c, μ is the global mean,
σ_c² is the variance in class c, and n_c is the class size.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FisherScore(Int32,Int32[])` | Creates a new instance of `FisherScore`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NFeaturesToSelect` | Gets the number of features to select. |
| `Scores` | Gets the computed Fisher Scores for each feature. |
| `SelectedIndices` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits Fisher Score by computing discriminative power for each feature. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets a boolean mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting Fisher-scored features. |

