---
title: "GradCAMExplainer<T>"
description: "Gradient-weighted Class Activation Mapping (Grad-CAM) explainer for CNNs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Gradient-weighted Class Activation Mapping (Grad-CAM) explainer for CNNs.

## For Beginners

Grad-CAM creates visual explanations showing which parts of an image
were most important for a CNN's prediction. It produces a heatmap highlighting important regions.

How it works:

1. Pass the image through the CNN
2. Get the feature maps from a convolutional layer (typically the last one)
3. Compute gradients of the target class score with respect to feature maps
4. Weight each feature map by its average gradient (importance)
5. Combine weighted feature maps and apply ReLU
6. Resize the result to match input image size

Why Grad-CAM is useful:

- Visual and intuitive: shows a heatmap over the image
- Class-discriminative: different classes highlight different regions
- Works with any CNN architecture (VGG, ResNet, etc.)
- No modification to the model architecture needed

Example: For an image classified as "cat":

- The heatmap would highlight the cat's face, body, ears
- Areas like the background would have low activation

Grad-CAM++ is an improved version that handles multiple instances of the same object better.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradCAMExplainer(Func<Tensor<>,Tensor<>>,Func<Tensor<>,Int32,Tensor<>>,Func<Tensor<>,Int32,Int32,Tensor<>>,Int32[],Int32[],Boolean)` | Initializes a new Grad-CAM explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradCAM(Tensor<>,Tensor<>)` | Computes standard Grad-CAM heatmap. |
| `ComputeGradCAMPlusPlus(Tensor<>,Tensor<>)` | Computes Grad-CAM++ heatmap (better for multiple instances). |
| `ComputeSimulatedHeatmap(Tensor<>,Int32)` | Computes a simulated heatmap when gradient access is not available. |
| `Explain(Vector<>)` | Computes Grad-CAM heatmap for an input image. |
| `ExplainBatch(Matrix<>)` |  |
| `ExplainTensor(Tensor<>,Int32,Int32)` | Computes Grad-CAM heatmap for an input tensor. |
| `GetPredictedClass(Tensor<>)` | Gets the predicted class index. |
| `NormalizeHeatmap([0:,0:])` | Normalizes heatmap to [0, 1] range. |
| `UpsampleHeatmap([0:,0:],Int32,Int32)` | Upsamples heatmap to target size using bilinear interpolation. |

