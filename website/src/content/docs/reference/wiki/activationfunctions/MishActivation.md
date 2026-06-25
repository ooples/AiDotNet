---
title: "MishActivation<T>"
description: "Implements the Mish activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Mish activation function for neural networks.

## For Beginners

The Mish activation function is a smooth, non-monotonic function that helps neural networks
learn complex patterns. It was introduced in 2019 and has shown good performance in many applications.

Mathematically, Mish is defined as: f(x) = x * tanh(softplus(x))
where softplus(x) = ln(1 + e^x)

Mish combines properties of several popular activation functions:

- It's smooth (no sharp corners like ReLU)
- It allows both positive and negative values (unlike ReLU which zeros out negatives)
- It's unbounded on the positive side (can output large positive values)
- It's bounded on the negative side (won't output extremely negative values)

These properties help neural networks learn more effectively in many situations.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether Mish supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Mish activation function to a single input value. |
| `Activate(Tensor<>)` | Applies Mish to a tensor via the engine so the gradient tape records the op. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative (gradient) of the Mish function for a single input value. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the Mish activation function on GPU. |
| `SupportsScalarOperations` | Determines if the activation function supports operations on individual scalar values. |

