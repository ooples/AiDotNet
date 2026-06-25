---
title: "RFE<T>"
description: "Recursive Feature Elimination for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Recursive Feature Elimination for feature selection.

## For Beginners

RFE is like an elimination tournament for features:

- Start with all features
- Remove the weakest performer each round
- Keep going until you have the desired number of features
- The surviving features are the most important ones

## How It Works

RFE performs feature selection by recursively removing features and building
a model on the remaining features. It ranks features by importance and removes
the least important ones until the desired number of features is reached.

The algorithm:

1. Train a model on all features and compute feature importances
2. Remove the least important feature(s)
3. Repeat until desired number of features is reached

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RFE(Int32,Int32,RFEImportanceMethod,Int32[])` | Creates a new instance of `RFE`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureImportances` | Gets the feature importances from the final model. |
| `NFeaturesToSelect` | Gets the number of features to select. |
| `Ranking` | Gets the feature ranking (1 = selected, 2+ = elimination order). |
| `SelectedIndices` | Gets the indices of selected features. |
| `Step` | Gets the step size (features removed per iteration). |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits RFE by recursively eliminating features. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets the support mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting the most important features. |

