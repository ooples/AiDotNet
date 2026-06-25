---
title: "ReLUActivation<T>"
description: "Implements the Rectified Linear Unit (ReLU) activation function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Rectified Linear Unit (ReLU) activation function.

## For Beginners

ReLU (Rectified Linear Unit) is like a filter that only lets positive values 
pass through unchanged, while changing all negative values to zero. Think of it as a function 
that "turns off" neurons that have negative values and keeps positive ones as they are.
This helps neural networks learn more effectively by introducing non-linearity.

## How It Works

The ReLU function is one of the most commonly used activation functions in neural networks.
It outputs the input directly if it is positive, otherwise, it outputs zero.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether ReLU supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the ReLU activation function to a single input value. |
| `Activate(Tensor<>)` | Applies the ReLU activation function to each element in a tensor. |
| `Activate(Vector<>)` | Applies the ReLU activation function to each element in a vector. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the ReLU function for a single input value. |
| `Derivative(Tensor<>)` | Calculates the derivative of the ReLU function for each element in a tensor. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of the ReLU function for a vector input. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the ReLU activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

