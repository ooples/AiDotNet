---
title: "HardTanhActivation<T>"
description: "Implements the Hard Tanh activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Hard Tanh activation function for neural networks.

## For Beginners

The Hard Tanh is a simplified version of the standard Tanh (hyperbolic tangent) function.

While the regular Tanh creates an S-shaped curve that smoothly transitions from -1 to 1,
the Hard Tanh uses straight lines to approximate this curve, making it:

1. Computationally faster (uses simple comparison operations instead of complex math)
2. Less smooth but still useful for many neural network applications

The function works like this:

- If input = -1: output = -1
- If input = 1: output = 1
- If -1 < input < 1: output = input (unchanged)

This creates a function that "clips" or "saturates" any input to the range [-1, 1],
with a straight line in between.

Hard Tanh is often used when you want the benefits of Tanh (centered around zero,
outputs between -1 and 1) but need faster computation, such as in deep networks
or resource-constrained environments.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether HardTanh supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Hard Tanh activation function to a single input value. |
| `Activate(Tensor<>)` | Applies HardTanh to a tensor via engine primitives so the gradient tape records every step. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the Hard Tanh function for a given input value. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the HardTanh activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function can operate on individual scalar values. |

