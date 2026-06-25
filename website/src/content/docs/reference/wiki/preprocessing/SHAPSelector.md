---
title: "SHAPSelector<T>"
description: "SHAP (SHapley Additive exPlanations) based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic`

SHAP (SHapley Additive exPlanations) based feature selection.

## For Beginners

SHAP answers: "How much did each feature contribute
to this prediction?"

Imagine a team of features making a prediction. SHAP figures out each
player's contribution fairly - not just by removing them, but by considering
every possible team combination they could have been in.

Features with high absolute SHAP values (positive or negative) are important.
SHAP provides both global importance and local explanations per sample.

## How It Works

SHAP uses Shapley values from game theory to explain feature importance.
Each feature's importance is its average contribution to predictions across
all possible feature combinations.

The algorithm (Kernel SHAP approximation):

1. Sample feature coalitions (subsets of features)
2. Weight coalitions by SHAP kernel
3. Predict with masked features (baseline substitution)
4. Solve weighted least squares for Shapley values

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SHAPSelector(Func<Matrix<>,Vector<>>,Int32,Int32,Int32,Nullable<Int32>,Int32[])` | Creates a new instance of `SHAPSelector`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NFeaturesToSelect` | Gets the number of features to select. |
| `SelectedIndices` | Gets the indices of selected features. |
| `ShapValues` | Gets the computed SHAP values (mean absolute). |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits SHAP selector by computing Shapley values for each feature. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets a boolean mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting SHAP-important features. |

