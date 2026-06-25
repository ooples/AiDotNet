---
title: "GuidedBackpropExplainer<T>"
description: "Guided Backpropagation explainer for neural network visualization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Guided Backpropagation explainer for neural network visualization.

## For Beginners

Guided Backpropagation is a gradient-based visualization technique
that shows which parts of an input most strongly activate a particular output.

**Key Insight:** Regular backpropagation can produce noisy gradients because it
propagates both positive and negative values through ReLU activations. Guided Backprop
only propagates gradients where BOTH the input to ReLU was positive AND the gradient
flowing back is positive.

**How it works:**

1. Do a forward pass to get activations
2. During backpropagation, at each ReLU:
- Regular backprop: gradient flows if input > 0
- Guided backprop: gradient flows if input > 0 AND gradient > 0
3. The result highlights features that positively contribute to the output

**Use cases:**

- Visualizing what a CNN "sees" in an image
- Understanding which pixels matter for a classification
- Debugging models (are they looking at the right features?)

**Compared to other methods:**

- Regular gradient: Noisy, can have negative attributions
- Guided backprop: Cleaner, only positive attributions
- DeconvNet: Only considers forward activations, not gradients
- GradCAM: Coarse localization (good with Guided Backprop = Guided GradCAM)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GuidedBackpropExplainer(Func<Vector<>,Vector<>>,Func<Tensor<>,Tensor<>>,Int32[])` | Initializes a Guided Backpropagation explainer from prediction functions. |
| `GuidedBackpropExplainer(INeuralNetwork<>,Int32[])` | Initializes a Guided Backpropagation explainer from a neural network. |

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
| `AiDotNet#Interfaces#ILocalExplainer{T,AiDotNet#Interpretability#Explainers#GuidedBackpropExplanation{T}}#Explain(Vector<>)` |  |
| `ComputeGuidedGradients(Vector<>,Nullable<Int32>)` | Computes guided gradients for an input. |
| `ComputeGuidedGradientsWithNetwork(Vector<>,Nullable<Int32>)` | Computes guided gradients using the neural network. |
| `ComputeNumericalGuidedGradients(Vector<>,Nullable<Int32>)` | Computes numerical guided gradients. |
| `Explain(Vector<>,Nullable<Int32>)` | Explains a single input using Guided Backpropagation. |
| `ExplainBatch(Matrix<>)` |  |
| `ExplainTensor(Tensor<>,Nullable<Int32>)` | Explains a tensor input (e.g., image). |
| `GetMultiIndex(Int32,Int32[])` | Converts linear index to multi-dimensional index. |
| `GetPredictedClass(Vector<>)` | Gets the predicted class index. |
| `GetPrediction(Vector<>)` | Gets predictions for an input. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |

