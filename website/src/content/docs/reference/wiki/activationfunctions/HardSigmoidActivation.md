---
title: "HardSigmoidActivation<T>"
description: "Implements the Hard Sigmoid activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Hard Sigmoid activation function for neural networks.

## For Beginners

The Hard Sigmoid is a simplified version of the standard Sigmoid function.

While the regular Sigmoid creates an S-shaped curve that smoothly transitions from 0 to 1,
the Hard Sigmoid uses straight lines to approximate this curve, making it:

1. Computationally faster (uses simple math operations instead of exponentials)
2. Less smooth but still useful for many neural network applications

The function works like this:

- If input = -1: output = 0
- If input = 1: output = 1
- If -1 < input < 1: output = (input + 1) / 2

This creates a straight line between (-1, 0) and (1, 1), with values clamped to the range [0, 1].

Hard Sigmoid is often used in mobile or embedded applications where computational
efficiency is important, or in certain types of recurrent neural networks.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether HardSigmoid supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Hard Sigmoid activation function to a single input value. |
| `Activate(Tensor<>)` | Applies HardSigmoid to a tensor via engine primitives so the gradient tape records every step. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the Hard Sigmoid function for a given input value. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the HardSigmoid activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function can operate on individual scalar values. |

