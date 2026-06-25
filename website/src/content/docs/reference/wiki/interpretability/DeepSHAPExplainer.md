---
title: "DeepSHAPExplainer<T>"
description: "DeepSHAP explainer combining GradientSHAP with DeepLIFT for efficient neural network explanations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

DeepSHAP explainer combining GradientSHAP with DeepLIFT for efficient neural network explanations.

## For Beginners

DeepSHAP is a fast method for computing SHAP values specifically designed
for deep neural networks. It combines two powerful ideas:

1. **DeepLIFT rules**: How to properly attribute through non-linearities (ReLU, etc.)
2. **Shapley sampling**: Using multiple baseline samples for better attribution

The key insight is that by using DeepLIFT's "multipliers" (instead of regular gradients),
we get attributions that are more stable and interpretable.

How DeepSHAP works:

1. Sample multiple reference inputs from your background data
2. For each reference, compute DeepLIFT-style attributions
3. Average the attributions across all references

This gives you Shapley-style attributions (fair credit assignment) computed
efficiently using backpropagation.

When to use DeepSHAP vs other methods:

- **DeepSHAP**: Best for deep neural networks, especially with ReLU activations
- **GradientSHAP**: Simpler, works well when gradients are reliable
- **KernelSHAP**: Model-agnostic but slower
- **IntegratedGradients**: Theoretically grounded but uses single baseline

DeepSHAP advantages:

- Fast (single backward pass per sample)
- Handles saturation regions well (via DeepLIFT rules)
- Produces Shapley-like fair attributions
- Works with any differentiable neural network

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepSHAPExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Vector<>>,Func<Vector<>,Vector<>,Vector<>>,Matrix<>,Int32,String[],Nullable<Int32>)` | Initializes a new DeepSHAP explainer with DeepLIFT multiplier support. |
| `DeepSHAPExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Vector<>>,Matrix<>,Int32,String[],Nullable<Int32>)` | Initializes a new DeepSHAP explainer. |
| `DeepSHAPExplainer(INeuralNetwork<>,Matrix<>,Int32,String[],Nullable<Int32>)` | Initializes a new DeepSHAP explainer from a neural network. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedValue` | Gets the expected (baseline) prediction value. |
| `IsGPUAccelerated` |  |
| `MethodName` |  |
| `NumSamples` | Gets the number of background samples used for computing attributions. |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDeepLIFTAttributions(Vector<>,Vector<>,Int32)` | Computes DeepLIFT-style attributions relative to a reference. |
| `ComputeExpectedValue(Matrix<>)` | Computes the expected prediction value from background data. |
| `Explain(Vector<>)` | Computes DeepSHAP attributions for an input. |
| `Explain(Vector<>,Int32)` | Computes DeepSHAP attributions for a specific output. |
| `ExplainBatch(Matrix<>)` |  |
| `ExplainGlobal(Matrix<>)` | Computes global feature importance by averaging absolute SHAP values. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |

