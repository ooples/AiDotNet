---
title: "LIMEExplainer<T>"
description: "Model-agnostic LIME (Local Interpretable Model-agnostic Explanations) explainer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Model-agnostic LIME (Local Interpretable Model-agnostic Explanations) explainer.

## For Beginners

LIME explains individual predictions by building a simple,
interpretable model (like linear regression) that approximates the complex model
locally around the prediction you want to understand.

How it works:

1. Generate perturbed samples near your instance (slightly modified versions)
2. Get the complex model's predictions for these samples
3. Fit a simple linear model to these nearby predictions
4. The linear model's coefficients show which features matter most

Think of it like zooming in on a curvy road - if you zoom in enough,
even a curved road looks straight. LIME zooms in on your prediction
and fits a "straight line" (linear model) to explain it.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LIMEExplainer(Func<Matrix<>,Vector<>>,Int32,Int32,Double,String[],Double[],Nullable<Int32>)` | Initializes a new LIME explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` |  |
| `MethodName` |  |
| `NumFeatures` | Gets the number of features being explained. |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeWeights(Matrix<>,Vector<>)` | Computes sample weights based on distance from original instance. |
| `Explain(Vector<>)` |  |
| `ExplainBatch(Matrix<>)` |  |
| `FitWeightedLinearModel(Matrix<>,Vector<>,Vector<>)` | Fits a weighted linear regression model. |
| `FromModel(IFullModel<,,>,Int32,Int32,Double,String[],Double[],Nullable<Int32>)` | Creates a LIME explainer from a model. |
| `GeneratePerturbedSamples(Vector<>,Random)` | Generates perturbed samples around the instance. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |

