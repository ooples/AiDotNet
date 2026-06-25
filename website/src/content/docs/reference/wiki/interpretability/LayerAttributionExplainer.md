---
title: "LayerAttributionExplainer<T>"
description: "Layer-level attribution explainer for computing attributions at intermediate layers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Layer-level attribution explainer for computing attributions at intermediate layers.
Supports LayerIntegratedGradients, LayerDeepLIFT, LayerGradientXActivation, and LayerConductance.

## For Beginners

While input attribution tells you which input features matter,
layer attribution tells you which NEURONS IN A HIDDEN LAYER matter for the output.

**Why layer attribution?**

- **Higher-level features:** Later layers encode abstract concepts, not raw pixels
- **Model understanding:** See what the model "thinks" at each stage
- **Debugging:** Find where information is lost or distorted
- **Transfer learning:** Understand which learned representations are being used

**Comparison of methods:**

- **LayerGradient:** Simple ∂output/∂layer, fast but noisy
- **LayerIntegratedGradients:** Path-integrated gradients, theoretically grounded
- **LayerDeepLIFT:** Difference from reference, handles saturation well
- **LayerConductance:** Layer activation × gradient, captures both magnitude and sensitivity

**Example:**
For an image classifier, layer attribution on the last conv layer might show
that specific feature maps (detecting eyes, fur, etc.) are important for the "cat" prediction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayerAttributionExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Vector<>>,Int32,Int32,LayerAttributionMethod,Int32)` | Initializes a layer attribution explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` |  |
| `MethodName` | Gets the method name. |
| `SupportsGlobalExplanations` | Gets whether this explainer supports global explanations. |
| `SupportsLocalExplanations` | Gets whether this explainer supports local explanations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeConductanceAttribution(Vector<>,Int32)` | Computes Layer Conductance attribution. |
| `ComputeDeepLIFTAttribution(Vector<>,Int32)` | Computes Layer DeepLIFT attribution. |
| `ComputeGradientAttribution(Vector<>,Int32)` | Computes simple gradient attribution. |
| `ComputeGradientXActivationAttribution(Vector<>,Int32,Vector<>)` | Computes Gradient × Activation attribution. |
| `ComputeIntegratedGradientsAttribution(Vector<>,Int32)` | Computes Layer Integrated Gradients attribution. |
| `ComputeLayerAttribution(Vector<>,Nullable<Int32>)` | Computes layer attribution for the specified layer. |
| `GetPredictedClass(Vector<>)` | Gets the predicted class from output. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |

