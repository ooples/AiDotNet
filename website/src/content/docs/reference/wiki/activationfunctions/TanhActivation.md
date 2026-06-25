---
title: "TanhActivation<T>"
description: "Implements the Hyperbolic Tangent (tanh) activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Hyperbolic Tangent (tanh) activation function for neural networks.

## For Beginners

The Hyperbolic Tangent (tanh) activation function is a popular choice in neural networks.
It transforms any input value to an output between -1 and 1, creating an S-shaped curve that's
symmetric around the origin.

Key properties of tanh:

- Outputs values between -1 and 1
- An input of 0 produces an output of 0
- Large positive inputs approach +1
- Large negative inputs approach -1
- It's zero-centered, which often helps with learning

When to use tanh:

- When you need outputs centered around zero
- For hidden layers in many types of neural networks
- When dealing with data that naturally has both positive and negative values

One limitation is the "vanishing gradient problem" - for very large or small inputs,
the function's slope becomes very small, which can slow down learning in deep networks.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether Tanh supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the tanh activation function to a single input value. |
| `Activate(Tensor<>)` | Applies the tanh activation function to a tensor of input values. |
| `Activate(Vector<>)` | Applies the tanh activation function to a vector of input values using SIMD optimization. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the tanh function for a single input value. |
| `Derivative(Tensor<>)` | Calculates the derivative of the tanh function for a tensor input. |
| `DerivativeFromOutput(Tensor<>)` | Computes tanh derivative given the post-activation output: 1 - y². |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the Tanh activation function on GPU. |
| `SupportsScalarOperations` | Indicates that this activation function supports operations on individual scalar values. |

