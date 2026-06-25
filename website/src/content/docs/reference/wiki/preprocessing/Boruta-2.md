---
title: "Boruta<T>"
description: "Boruta feature selection algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Boruta feature selection algorithm.

## For Beginners

Boruta asks: "Is this feature better than random noise?"

It creates "shadow" features by shuffling your real features randomly.
If your original feature has higher importance than these random shadows,
it's probably useful. If it's worse than random noise, it's rejected.

Unlike SelectKBest which picks the "top K", Boruta finds ALL features
that are genuinely useful, which could be 3 or 30 depending on your data.

## How It Works

Boruta is an all-relevant feature selection method that uses shadow features
(randomized copies of original features) as a benchmark for importance.

The algorithm:

1. Create shadow features by shuffling original features
2. Train a model on [original + shadow] features
3. Compare each feature's importance to the max shadow importance
4. Features consistently beating shadows are "confirmed"
5. Features consistently losing to shadows are "rejected"
6. Repeat until all features are decided or max iterations reached

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Boruta(Func<Matrix<>,Vector<>,Double[]>,Int32,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `Boruta`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Decisions` | Gets the decisions for each feature. |
| `Iterations` | Gets the number of iterations performed. |
| `MeanImportances` | Gets the mean importance scores for each feature. |
| `SelectedIndices` | Gets the indices of confirmed features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits Boruta by comparing feature importances to shadow features. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets a boolean mask indicating which features are confirmed. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting confirmed features. |

