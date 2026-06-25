---
title: "SigmoidActivation<T>"
description: "Implements the Sigmoid activation function, one of the most common activation functions in neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Sigmoid activation function, one of the most common activation functions in neural networks.

## For Beginners

The Sigmoid function is like a "squashing" function that takes any number (from negative
infinity to positive infinity) and converts it to a value between 0 and 1. This is useful in neural networks
because it helps transform unbounded values into a probability-like range. The function creates an S-shaped
curve that approaches 0 for very negative inputs and approaches 1 for very positive inputs, with a smooth
transition in between. However, one limitation is that for extreme values (very large positive or negative),
the gradient becomes very small, which can slow down learning in deep networks (known as the "vanishing
gradient problem").

## How It Works

The Sigmoid function maps any input value to an output between 0 and 1, creating an S-shaped curve.
It's often used in the output layer of binary classification problems or in hidden layers of neural networks.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether Sigmoid supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Sigmoid activation function to a single input value. |
| `Activate(Tensor<>)` | Applies the Sigmoid activation function to each element in a tensor. |
| `Activate(Vector<>)` | Applies the Sigmoid activation function to each element in a vector using SIMD optimization. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the Sigmoid function for a single input value. |
| `Derivative(Tensor<>)` | Calculates the derivative of the Sigmoid function for each element in a tensor. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of the Sigmoid function for a vector input. |
| `DerivativeFromOutput(Tensor<>)` | Computes sigmoid derivative given the post-activation output: y * (1 - y). |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the Sigmoid activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

