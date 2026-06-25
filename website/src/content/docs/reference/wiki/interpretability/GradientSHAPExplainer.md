---
title: "GradientSHAPExplainer<T>"
description: "GradientSHAP explainer - a faster approximation of SHAP using gradients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

GradientSHAP explainer - a faster approximation of SHAP using gradients.

## For Beginners

GradientSHAP combines ideas from Integrated Gradients and SHAP
to create a faster method for explaining neural networks.

How it works:

1. Sample random baselines from your background data
2. For each baseline, compute something like Integrated Gradients
3. Average the results to get SHAP-like values

Comparison with other methods:

- **KernelSHAP**: Model-agnostic but slow (doesn't use gradients)
- **DeepSHAP**: Fast but requires access to layer activations
- **GradientSHAP**: Good balance - uses gradients but doesn't need internal model access

Why use GradientSHAP?

- Much faster than KernelSHAP for neural networks
- Only requires gradient computation (not layer access)
- Approximates SHAP values reasonably well
- Better than plain gradients (uses baselines for context)

When to use:

- You have a neural network
- KernelSHAP is too slow
- You don't have access to model internals (for DeepSHAP)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientSHAPExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Vector<>>,Matrix<>,Int32,Int32,Boolean,Double,String[],Nullable<Int32>)` | Initializes a new GradientSHAP explainer. |

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
| `ComputeGradient(Vector<>,Int32)` | Computes gradient using provided function or numerical approximation. |
| `ComputeIntegratedGradients(Vector<>,Vector<>,Int32,Random)` | Computes Integrated Gradients from baseline to input. |
| `ComputeNumericalGradient(Vector<>,Int32)` | Computes numerical gradient. |
| `Explain(Vector<>)` | Computes GradientSHAP attributions for an input. |
| `Explain(Vector<>,Int32)` | Computes GradientSHAP attributions for a specific output. |
| `ExplainBatch(Matrix<>)` |  |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |

