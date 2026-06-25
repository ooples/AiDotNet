---
title: "LIMESelector<T>"
description: "LIME (Local Interpretable Model-agnostic Explanations) based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic`

LIME (Local Interpretable Model-agnostic Explanations) based feature selection.

## For Beginners

LIME answers: "For this specific prediction,
which features mattered most?"

It works by creating slightly modified versions of your data point,
seeing how predictions change, and fitting a simple linear model to
understand which features drive the prediction locally.

Unlike global methods, LIME provides instance-specific explanations.
For feature selection, we average importance across many instances.

## How It Works

LIME explains individual predictions by approximating the model locally with
an interpretable linear model. Feature importance is derived from the linear
coefficients of these local explanations.

The algorithm:

1. For each instance to explain:

a. Generate perturbed samples around the instance
b. Weight samples by proximity to original instance
c. Fit weighted linear regression on perturbed samples
d. Extract feature importance from coefficients

2. Aggregate importance across all explained instances

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LIMESelector(Func<Matrix<>,Vector<>>,Int32,Int32,Int32,Double,Nullable<Int32>,Int32[])` | Creates a new instance of `LIMESelector`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Importances` | Gets the computed LIME importance values (mean absolute coefficients). |
| `NFeaturesToSelect` | Gets the number of features to select. |
| `SelectedIndices` | Gets the indices of selected features. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits LIME selector by computing local explanations and aggregating importance. |
| `FitCore(Matrix<>)` | Fits the selector (requires target via specialized Fit method). |
| `FitTransform(Matrix<>,Vector<>)` | Fits and transforms the data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets a boolean mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Transforms the data by selecting LIME-important features. |

