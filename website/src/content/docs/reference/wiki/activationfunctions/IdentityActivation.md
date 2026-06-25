---
title: "IdentityActivation<T>"
description: "Implements the Identity activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Identity activation function for neural networks.

## For Beginners

The Identity activation function is the simplest activation function - it returns exactly what you give it.

When you pass a value through this function:

- Input of 2 ? Output of 2
- Input of -3.5 ? Output of -3.5
- And so on...

Think of it like a straight line on a graph where y = x.

While most neural networks use non-linear activation functions (like ReLU or Sigmoid) to model complex patterns,
the Identity function can be useful in:

- The output layer of regression problems (when predicting continuous values)
- Testing or debugging neural networks
- Certain network architectures where you want values to pass through unchanged

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Identity activation function to a single value. |
| `Activate(Tensor<>)` | Returns the input tensor unchanged — no allocation, preserves tape chain. |
| `Activate(Vector<>)` | Applies the Identity activation function to a vector of values. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative (gradient) of the Identity function for a single value. |
| `Derivative(Vector<>)` | Calculates the derivative (gradient) of the Identity function for a vector of values. |
| `SupportsScalarOperations` | Indicates whether this activation function can operate on individual scalar values. |

