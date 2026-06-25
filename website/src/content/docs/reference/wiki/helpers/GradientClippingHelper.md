---
title: "GradientClippingHelper"
description: "Provides gradient clipping utilities to prevent exploding gradients during training."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides gradient clipping utilities to prevent exploding gradients during training.

## For Beginners

During neural network training, gradients tell us how to adjust
weights. Sometimes gradients become extremely large ("exploding gradients"), which can
destabilize training. Gradient clipping limits the magnitude of gradients to keep
training stable.

There are two main approaches:

- **Clip by Value**: Limits each gradient element to a range (e.g., -1 to 1)
- **Clip by Norm**: Scales the entire gradient vector if its norm exceeds a threshold

The "by norm" approach is generally preferred as it preserves gradient direction.

## Methods

| Method | Summary |
|:-----|:--------|
| `AreGradientsExploding(Vector<>,Double)` | Detects if gradients are exploding (have very large values). |
| `AreGradientsVanishing(Vector<>,Double)` | Detects if gradients are vanishing (have very small values). |
| `ClipAdaptive(Vector<>,Vector<>,Double)` | Applies adaptive gradient clipping based on parameter norm. |
| `ClipByGlobalNorm(List<Vector<>>,Double)` | Clips gradients by global norm across multiple gradient vectors. |
| `ClipByNorm(Tensor<>,Double)` | Clips tensor gradients by their L2 norm. |
| `ClipByNorm(Vector<>,Double)` | Clips gradients by their L2 norm (global norm clipping). |
| `ClipByNormInPlace(Vector<>,Double)` | Clips gradients by their L2 norm in place. |
| `ClipByValue(Vector<>,Double)` | Clips gradient values to a specified range [-maxValue, maxValue]. |
| `ClipByValueInPlace(Vector<>,Double)` | Clips gradient values to a specified range [-maxValue, maxValue] in place. |
| `ComputeGlobalNorm(List<Vector<>>)` | Computes the global L2 norm across multiple gradient vectors. |
| `ComputeNorm(Vector<>)` | Computes the L2 norm of a gradient vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultMaxNorm` | Default maximum gradient norm for clipping. |
| `DefaultMaxValue` | Default maximum gradient value for value clipping. |

