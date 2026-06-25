---
title: "SwishActivation<T>"
description: "Implements the Swish activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Swish activation function for neural networks.

## For Beginners

Swish is a newer activation function developed by researchers at Google.
Its formula is x * sigmoid(x), which means it multiplies the input by the sigmoid of the input.

Key characteristics of Swish include:

- It's smooth everywhere, unlike ReLU which has a sharp corner at x=0
- It allows some negative values through, which can help with learning
- It behaves somewhat like ReLU for positive values, but has a smoother transition
- It has been shown to outperform ReLU in some deep neural networks

Swish combines some of the best properties of ReLU and sigmoid functions,
making it effective for many deep learning applications.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether Swish supports GPU-resident training. |
| `SupportsJitCompilation` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Swish activation function to a scalar input value. |
| `Activate(Tensor<>)` | Applies Swish to a tensor via the engine so the gradient tape records the op. |
| `Activate(Vector<>)` | Applies the Swish activation function to each element of an input vector. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the Swish activation function for a scalar input value. |
| `Derivative(Vector<>)` | Calculates the derivative of the Swish activation function for each element of an input vector. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the Swish activation function on GPU. |
| `Sigmoid()` | Calculates the sigmoid function for a scalar value. |
| `SupportsScalarOperations` | Determines if the activation function supports operations on individual scalar values. |

