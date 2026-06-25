---
title: "SoftPlusActivation<T>"
description: "Implements the SoftPlus activation function, which is a smooth approximation of the ReLU function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the SoftPlus activation function, which is a smooth approximation of the ReLU function.

## For Beginners

SoftPlus is like a "softer" version of the popular ReLU activation function. 
While ReLU outputs exactly 0 for any negative input and keeps positive values unchanged,
SoftPlus creates a smooth curve that's very close to ReLU but without the sharp corner at x=0.

For negative inputs, SoftPlus outputs small positive values (approaching 0).
For large positive inputs, SoftPlus outputs values very close to the input itself.

This smoothness can be helpful in some neural networks because it means the function is differentiable
everywhere (it has a well-defined slope at every point), which can make training more stable.

## How It Works

The SoftPlus function is defined as: f(x) = ln(1 + e^x), where ln is the natural logarithm.
It produces output that is always positive and approaches the ReLU function (max(0,x)) but with a smooth transition at x=0.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether SoftPlus supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the SoftPlus activation function to a single input value. |
| `Activate(Tensor<>)` | Applies the SoftPlus activation function to each element in a tensor. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the SoftPlus function for a given input. |
| `Derivative(Tensor<>)` | Calculates the derivative of the SoftPlus function for each element in a tensor. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the SoftPlus activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

