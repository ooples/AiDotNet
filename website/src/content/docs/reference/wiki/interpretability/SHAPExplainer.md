---
title: "SHAPExplainer<T>"
description: "Model-agnostic SHAP (SHapley Additive exPlanations) explainer using Kernel SHAP algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Model-agnostic SHAP (SHapley Additive exPlanations) explainer using Kernel SHAP algorithm.

## For Beginners

SHAP values come from game theory and answer the question:
"How much did each feature contribute to this specific prediction?"

Imagine you're splitting a restaurant bill fairly among friends based on what each person ordered.
SHAP does something similar - it fairly distributes the "credit" for a prediction among all input features.

Key properties of SHAP values:

- They sum up to the difference between the prediction and the average prediction
- Positive values mean the feature pushed the prediction higher
- Negative values mean the feature pushed the prediction lower
- The magnitude shows how important that feature was

This implementation uses Kernel SHAP, which works with ANY model by treating it as a black box.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SHAPExplainer(Func<Matrix<>,Vector<>>,Matrix<>,Int32,String[],Nullable<Int32>)` | Initializes a new SHAP explainer with a prediction function and background data. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaselineValue` | Gets the baseline (expected) prediction value computed from background data. |
| `IsGPUAccelerated` |  |
| `MethodName` |  |
| `NumFeatures` | Gets the number of features being explained. |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCoalitionPrediction(Vector<>,Boolean[])` | Computes the expected prediction for a coalition by marginalizing over background data. |
| `ComputeKernelSHAP(Vector<>)` | Computes SHAP values using the Kernel SHAP algorithm. |
| `Explain(Vector<>)` |  |
| `ExplainBatch(Matrix<>)` |  |
| `ExplainGlobal(Matrix<>)` |  |
| `FromModel(IFullModel<,,>,Matrix<>,Int32,String[],Nullable<Int32>)` | Initializes a new SHAP explainer with a model that implements prediction. |
| `LogBinomial(Int32,Int32)` | Computes the natural logarithm of the binomial coefficient C(n, k). |
| `LogFactorial(Int32)` | Computes log(n!) using a simple approach for reasonable n values. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |
| `SolveLinearSystem(Double[0:,0:],Double[])` | Solves a linear system Ax = b using Gaussian elimination with partial pivoting. |
| `SolveWeightedLeastSquares(List<Boolean[]>,List<>,List<Double>,Int32)` | Solves weighted least squares to compute SHAP values from coalition predictions. |
| `SolveWeightedLeastSquaresFromVector(List<Boolean[]>,Vector<>,List<Double>,Int32)` | Solves weighted least squares from a predictions vector. |

