---
title: "ActivationFunctionBase<T>"
description: "Base class for all activation functions used in neural networks."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ActivationFunctions`

Base class for all activation functions used in neural networks.

## For Beginners

Activation functions are mathematical operations that determine the output
of a neural network node. They introduce non-linearity into the network, allowing it to
learn complex patterns. Think of them as decision-makers that determine how strongly a
neuron "fires" based on its inputs.

Common activation functions include:

- Sigmoid: Outputs values between 0 and 1 (like probabilities)
- ReLU: Returns 0 for negative inputs, or the input value for positive inputs
- Tanh: Similar to sigmoid but outputs values between -1 and 1

The "derivative" methods are used during training to determine how to adjust the network's
weights to improve its accuracy.

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vector operations. |
| `SupportsGpuTraining` | Gets whether this activation function supports GPU-resident training. |
| `SupportsJitCompilation` | Gets whether this activation function supports JIT compilation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the activation function to a single input value. |
| `Activate(Tensor<>)` | Applies the activation function to each element in a tensor. |
| `Activate(Vector<>)` | Applies the activation function to each element in a vector. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Backward(Tensor<>,Tensor<>)` | Calculates the backward pass gradient for this activation function. |
| `Derivative()` | Calculates the derivative of the activation function for a single input value. |
| `Derivative(Tensor<>)` | Calculates the derivative for each element in a tensor. |
| `Derivative(Vector<>)` | Calculates the derivative matrix for a vector input. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the activation function on GPU. |
| `SupportsScalarOperations` | Determines if the activation function supports operations on individual scalar values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides mathematical operations for the numeric type T. |

