---
title: "GuidedGradCAMExplainer<T>"
description: "Guided GradCAM explainer combining GuidedBackprop with GradCAM for high-resolution explanations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Guided GradCAM explainer combining GuidedBackprop with GradCAM for high-resolution explanations.

## For Beginners

Guided GradCAM is the best of both worlds - it combines:

1. **GradCAM:** Tells you WHERE to look (coarse localization)
2. **GuidedBackprop:** Tells you WHAT to look for (fine-grained details)

**How it works:**

1. Compute GradCAM map (coarse localization of important regions)
2. Upsample GradCAM to input resolution
3. Compute Guided Backpropagation (fine-grained pixel-level importance)
4. Element-wise multiply: GuidedGradCAM = GuidedBackprop * upsampled_GradCAM

**Why this works:**

- GradCAM alone is too coarse (can't see fine details)
- GuidedBackprop alone highlights details everywhere (not localized)
- Multiplying them together keeps only important details in important regions

**Result:**
High-resolution, class-discriminative visualizations that show both WHERE
the model is looking AND WHAT features it sees there.

**Use cases:**

- Medical imaging (showing exactly what pixels indicate disease)
- Object detection debugging (why was this object detected?)
- Fine-grained classification (what distinguishes this bird species?)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GuidedGradCAMExplainer(GuidedBackpropExplainer<>,LayerGradCAMExplainer<>,Int32[])` | Initializes a Guided GradCAM explainer from component explainers. |
| `GuidedGradCAMExplainer(INeuralNetwork<>,Func<Vector<>,Vector<>>,Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Vector<>>,Int32[],Int32,Int32,Int32)` | Initializes a Guided GradCAM explainer from a neural network. |

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
| `AiDotNet#Interfaces#ILocalExplainer{T,AiDotNet#Interpretability#Explainers#GuidedGradCAMExplanation{T}}#Explain(Vector<>)` |  |
| `ComputeGuidedGradCAM(Vector<>,Tensor<>)` | Computes Guided GradCAM by element-wise multiplication. |
| `Explain(Vector<>,Nullable<Int32>)` | Generates a Guided GradCAM explanation for an input. |
| `ExplainBatch(Matrix<>)` |  |
| `ExplainTensor(Tensor<>,Nullable<Int32>)` | Explains a tensor input. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |
| `UpsampleGradCAM(Matrix<>)` | Upsamples GradCAM to input resolution. |

