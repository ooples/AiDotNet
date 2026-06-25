---
title: "IConvolutionalNetwork<T, TInput, TOutput>"
description: "Interface for convolutional neural networks that support Grad-CAM explanation."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interpretability.Interfaces`

Interface for convolutional neural networks that support Grad-CAM explanation.

## For Beginners

Grad-CAM creates visual explanations showing which parts of
an image the CNN focused on. This interface provides the methods needed to extract
feature maps (what the CNN "sees") and their gradients (what matters for the prediction).

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFeatureMapsAndGradients(Tensor<>,Int32)` | Gets feature maps and their gradients from the last convolutional layer. |

