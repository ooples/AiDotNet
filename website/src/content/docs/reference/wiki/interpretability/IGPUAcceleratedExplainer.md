---
title: "IGPUAcceleratedExplainer<T>"
description: "Interface for explainers that support GPU acceleration."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interpretability.Helpers`

Interface for explainers that support GPU acceleration.

## For Beginners

Explainers implementing this interface can leverage GPU
hardware for faster computation. The GPU helper is optional - if not provided,
the explainer falls back to CPU computation.

To enable GPU acceleration:

1. Create a GPUExplainerHelper with your GPU runtime
2. Pass it to the explainer via SetGPUHelper()
3. Explanations will automatically use GPU when beneficial

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` | Gets whether GPU acceleration is currently enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SetGPUHelper(GPUExplainerHelper<>)` | Sets the GPU helper for accelerated computation. |

