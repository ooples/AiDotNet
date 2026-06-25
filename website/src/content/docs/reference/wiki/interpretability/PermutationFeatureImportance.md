---
title: "PermutationFeatureImportance<T>"
description: "Model-agnostic Permutation Feature Importance calculator."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Model-agnostic Permutation Feature Importance calculator.

## For Beginners

Permutation Feature Importance measures how important each feature
is by randomly shuffling that feature's values and measuring how much worse the model performs.

The intuition is simple:

- If a feature is important, shuffling it destroys valuable information, and the model performs worse
- If a feature isn't important, shuffling it doesn't matter much

This method works with ANY model because it only looks at the model's predictions,
not its internal structure. It's like testing which ingredient is most important in a recipe
by randomly swapping each ingredient and seeing how much the dish changes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PermutationFeatureImportance(Func<Matrix<>,Vector<>>,Func<Vector<>,Vector<>,>,Int32,String[],Nullable<Int32>)` | Initializes a new Permutation Feature Importance calculator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` |  |
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Matrix<>,Vector<>)` | Calculates permutation feature importance. |
| `ExplainGlobal(Matrix<>)` |  |
| `ForClassification(IFullModel<,,>,Int32,String[],Nullable<Int32>)` | Creates a calculator using accuracy as the scoring function (for classification). |
| `ForRegression(IFullModel<,,>,Int32,String[],Nullable<Int32>)` | Creates a calculator using R² as the scoring function (for regression). |
| `FromModel(IFullModel<,,>,Func<Vector<>,Vector<>,>,Int32,String[],Nullable<Int32>)` | Creates a Permutation Feature Importance calculator from a model. |
| `PermuteColumn(Matrix<>,Int32,Random)` | Permutes (shuffles) a column in the matrix. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |

