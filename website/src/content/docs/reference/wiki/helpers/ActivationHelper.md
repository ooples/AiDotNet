---
title: "ActivationHelper"
description: "Provides centralized helper methods for applying activation functions with optimal performance."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides centralized helper methods for applying activation functions with optimal performance.
Uses Engine methods (GPU/SIMD) for known activation types, falls back to standard activation otherwise.

## How It Works

This class consolidates activation type-checking logic in one place, following DRY (Don't Repeat Yourself)
and SOLID principles. All layers should use these methods instead of duplicating if-else chains.

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivation(IActivationFunction<>,Tensor<>,IEngine)` | Applies a scalar (element-wise) activation function to a tensor, routing known activations through tape-connected Engine kernels and falling back to the activation's own `Tensor{` for custom types. |
| `ApplyActivation(IVectorActivationFunction<>,Tensor<>,IEngine)` | Applies a vector activation function to a tensor using Engine methods when possible. |
| `ApplyActivation(IVectorActivationFunction<>,Vector<>,IEngine)` | Applies a vector activation function to a vector using Engine methods when possible. |

