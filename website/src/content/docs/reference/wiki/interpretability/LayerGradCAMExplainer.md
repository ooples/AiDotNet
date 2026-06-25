---
title: "LayerGradCAMExplainer<T>"
description: "Layer GradCAM (Gradient-weighted Class Activation Mapping) explainer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Layer GradCAM (Gradient-weighted Class Activation Mapping) explainer.

## For Beginners

GradCAM produces a coarse localization map showing which regions
of an input (usually an image) are important for a prediction.

**How it works:**

1. Get activations at a target layer (usually the last convolutional layer)
2. Compute gradients of the target class with respect to these activations
3. Global average pool the gradients to get importance weights for each channel
4. Compute weighted combination of activation channels
5. Apply ReLU to keep only positive influences

**Why GradCAM is useful:**

- Shows WHERE the model is looking, not just WHAT features matter
- Works with any CNN architecture
- Produces interpretable heatmaps
- Doesn't require architectural changes or retraining

**Layer choice:**

- Last conv layer: Best balance of semantic meaning and spatial detail
- Earlier layers: More spatial detail but less semantic meaning
- Later layers: More semantic but coarser resolution

**Limitations:**

- Resolution limited by the target layer's spatial dimensions
- May miss fine-grained details (use GuidedGradCAM for that)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayerGradCAMExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Vector<>>,Int32,Int32,Int32,Int32[])` | Initializes a LayerGradCAM explainer. |
| `LayerGradCAMExplainer(INeuralNetwork<>,Int32,Int32,Int32,Int32,Int32[])` | Initializes a LayerGradCAM explainer from a neural network. |

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
| `AiDotNet#Interfaces#ILocalExplainer{T,AiDotNet#Interpretability#Explainers#LayerGradCAMExplanation{T}}#Explain(Vector<>)` |  |
| `ComputeGradCAM(Vector<>,Vector<>)` | Computes the GradCAM activation map. |
| `ComputeNumericalLayerGradients(Vector<>,Int32)` | Computes numerical gradients for the layer. |
| `Explain(Vector<>,Nullable<Int32>)` | Generates a GradCAM explanation for an input. |
| `ExplainBatch(Matrix<>)` |  |
| `GetLayerActivations(Vector<>)` | Gets layer activations. |
| `GetLayerGradients(Vector<>,Int32)` | Gets layer gradients. |
| `GetPredictedClass(Vector<>)` | Gets the predicted class. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |
| `UpsampleGradCAM(Matrix<>)` | Upsamples the GradCAM map to input size using bilinear interpolation. |

