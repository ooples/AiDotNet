---
title: "InputGradientHelper<T>"
description: "Provides unified input gradient computation for interpretability explainers."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Interpretability.Helpers`

Provides unified input gradient computation for interpretability explainers.

## For Beginners

This helper computes how the model's output changes when you
slightly change each input feature. These "input gradients" are essential for many
interpretability methods like Integrated Gradients, GradCAM, and DeepLIFT.

The helper automatically chooses the best available method:

1. **Native Backpropagation**: If the model is a neural network with backprop support,

uses the efficient built-in gradient computation.

2. **GradientTape**: If the model uses the autodiff system with TensorOperations,

uses tape-based automatic differentiation.

3. **Numerical Gradients**: As a fallback, computes approximate gradients by

slightly perturbing each input (slower but works with any model).

Why input gradients matter:

- They show the sensitivity of the output to each input feature
- Positive gradient: increasing this feature increases the output
- Negative gradient: increasing this feature decreases the output
- Large magnitude: the feature has strong influence on the output

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InputGradientHelper(Func<Tensor<>,Tensor<>>,Double)` | Creates a gradient helper from a tensor prediction function. |
| `InputGradientHelper(Func<Vector<>,Vector<>>,Double)` | Creates a gradient helper from a prediction function (uses numerical gradients). |
| `InputGradientHelper(INeuralNetwork<>,Double)` | Creates a gradient helper from a neural network model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GradientMethod` | Gets the method name used for gradient computation. |
| `SupportsExactGradients` | Gets whether this helper can compute exact gradients (via backprop or autodiff). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Int32)` | Computes gradients of the output with respect to the input. |
| `ComputeGradientTensor(Tensor<>,Int32)` | Computes gradients for a tensor input. |
| `ComputeGradientViaBackprop(Vector<>,Int32)` | Computes gradients via neural network backpropagation. |
| `ComputeIntegratedGradients(Vector<>,Vector<>,Int32,Int32)` | Computes Integrated Gradients attributions directly. |
| `ComputeNumericalGradient(Vector<>,Int32)` | Computes numerical gradient approximation for vector input. |
| `ComputeNumericalGradientTensor(Tensor<>,Int32)` | Computes numerical gradient approximation for tensor input. |
| `ComputePathGradients(Vector<>,Vector<>,Int32,Int32)` | Computes gradients along a path from baseline to input (for Integrated Gradients). |
| `CreateGradientFunction` | Creates a gradient function suitable for Integrated Gradients or similar methods. |
| `CreateTensorGradientFunction` | Creates a tensor gradient function for image-based explainers. |

