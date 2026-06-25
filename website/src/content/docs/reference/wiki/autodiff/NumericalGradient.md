---
title: "NumericalGradient<T>"
description: "Provides numerical gradient computation using finite differences for gradient verification."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Autodiff.Testing`

Provides numerical gradient computation using finite differences for gradient verification.

## For Beginners

This class helps verify that our gradient calculations are correct.

The idea is simple:

1. We want to know how much f(x) changes when we change x slightly
2. We compute f(x+h) and f(x-h) where h is a tiny number
3. The gradient is approximately: (f(x+h) - f(x-h)) / (2h)

This is called the "central difference" method. It's slow but reliable.
We use it to check that our fast autodiff gradients are correct.

Example:

- For f(x) = x^2, the true gradient is 2x
- At x=3: numerical gradient = ((3+h)^2 - (3-h)^2) / (2h) ≈ 6
- Autodiff should also give 6

## How It Works

This utility class computes gradients numerically using the central difference method.
It serves as a ground truth for verifying that automatic differentiation produces correct gradients.

## Methods

| Method | Summary |
|:-----|:--------|
| `Compare(Tensor<>,Tensor<>,Double,Double)` | Compares two tensors and returns the maximum relative error. |
| `ComputeForBinaryOperation(Tensor<>,Tensor<>,Tensor<>,Func<ComputationNode<>,ComputationNode<>,ComputationNode<>>,Double)` | Computes numerical gradient for a binary operation (two inputs). |
| `ComputeForOperation(Tensor<>,Tensor<>,Func<ComputationNode<>,ComputationNode<>>,Double)` | Computes numerical gradient using ComputationNode operations for direct comparison with autodiff. |
| `ComputeForScalarFunction(Tensor<>,Func<Tensor<>,>,Double)` | Computes numerical gradient for a scalar-valued function of a tensor. |
| `ComputeForTensorFunction(Tensor<>,Tensor<>,Func<Tensor<>,Tensor<>>,Double)` | Computes numerical gradient for a tensor-valued function, given an output gradient. |
| `ComputeRelativeError(Double,Double)` | Computes relative error between two values. |
| `DotProduct(Tensor<>,Tensor<>)` | Computes dot product of two tensors (sum of element-wise products). |
| `FormatShape(Int32[])` | Formats a shape array for display. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | The numeric operations appropriate for the generic type T. |

