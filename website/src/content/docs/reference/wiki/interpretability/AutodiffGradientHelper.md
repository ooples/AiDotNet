---
title: "AutodiffGradientHelper<T>"
description: "Provides gradient computation using the GradientTape automatic differentiation system."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Interpretability.Helpers`

Provides gradient computation using the GradientTape automatic differentiation system.

## For Beginners

This helper uses automatic differentiation (autodiff) to compute
gradients. Autodiff is a technique that records mathematical operations as they happen
and then plays them backward to compute gradients.

How it works:

1. Create a GradientTape and start recording
2. Mark the input as a variable we want gradients for (Watch)
3. Run the model's forward pass - all operations are recorded
4. Call Gradient to compute how the output changes with respect to the input

This is the same technology used by TensorFlow and PyTorch for training neural networks.

When to use this helper:

- When your model is built using TensorOperations (autodiff-aware operations)
- When you need gradients with respect to multiple variables
- When you want to compute higher-order gradients (gradient of gradient)

This helper integrates with the existing GradientTape system in AiDotNet.Autodiff.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutodiffGradientHelper(Func<ComputationNode<>,ComputationNode<>>)` | Creates an autodiff gradient helper from a model function that uses ComputationNodes. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GradientMethod` | Gets the method name for this gradient computation approach. |
| `SupportsExactGradients` | Gets whether this helper supports exact gradients. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Tensor<>,Int32)` | Computes gradients for a tensor input. |
| `ComputeGradient(Vector<>,Int32)` | Computes gradients for a vector input. |
| `ComputeHessianVectorProduct(Tensor<>,Tensor<>,Int32)` | Computes second-order gradients (Hessian-vector product). |
| `CreateGradientFunction` | Creates a gradient function suitable for explainers. |
| `CreateTensorGradientFunction` | Creates a tensor gradient function for image-based explainers. |

