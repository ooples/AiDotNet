---
title: "GlobalSurrogateExplainer<T>"
description: "Global Surrogate Model explainer that approximates a complex model with an interpretable one."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Global Surrogate Model explainer that approximates a complex model with an interpretable one.

## For Beginners

A global surrogate model is a simple, interpretable model (like
linear regression or a decision tree) that tries to mimic a complex "black box" model.

How it works:

1. Use the complex model to make predictions on a dataset
2. Train a simple model to predict what the complex model predicts
3. Analyze the simple model to understand the complex one

Think of it like having a translator explain a complex foreign language document.
The translator (surrogate model) isn't perfect, but they can explain the main ideas
in a way you understand.

The R² score tells you how well the surrogate approximates the black box:

- R² close to 1.0 = surrogate explains the black box well
- R² far from 1.0 = surrogate is too simple to capture the black box behavior

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GlobalSurrogateExplainer(Func<Matrix<>,Vector<>>,Int32,String[])` | Initializes a new Global Surrogate explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Fidelity` | Gets the fidelity (R²) of the surrogate model. |
| `IsFitted` | Gets whether the surrogate model has been fitted. |
| `MethodName` |  |
| `NumFeatures` | Gets the number of features. |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compare(Matrix<>)` | Compares the surrogate predictions with the black box predictions. |
| `ExplainGlobal(Matrix<>)` |  |
| `Fit(Matrix<>)` | Fits the surrogate model to approximate the black box on the given data. |
| `FitLinearSurrogate(Matrix<>,Vector<>)` | Fits a linear regression model to approximate the black box. |
| `FromModel(IFullModel<,,>,Int32,String[])` | Creates a Global Surrogate explainer from a model. |
| `PredictSurrogate(Matrix<>)` | Uses the surrogate model to make predictions. |

