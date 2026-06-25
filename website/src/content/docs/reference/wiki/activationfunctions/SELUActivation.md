---
title: "SELUActivation<T>"
description: "Implements the Scaled Exponential Linear Unit (SELU) activation function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Scaled Exponential Linear Unit (SELU) activation function.

## For Beginners

SELU (Scaled Exponential Linear Unit) is a special activation function that 
helps neural networks train more effectively. Unlike simpler functions like ReLU, SELU has 
carefully chosen constants (alpha and lambda) that help keep the data flowing through your 
neural network well-balanced. This means your network can learn faster and more reliably 
without requiring extra normalization steps. Think of it as a self-regulating function that 
keeps your data in a "sweet spot" range as it passes through the network.

## How It Works

SELU is an activation function that enables self-normalizing properties in neural networks.
It automatically ensures that the outputs maintain a mean of 0 and standard deviation of 1 
across the network, which helps with training stability and convergence.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SELUActivation` | Initializes a new instance of the SELU activation function with predefined constants. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether SELU supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the SELU activation function to a single input value. |
| `Activate(Tensor<>)` | Applies SELU to a tensor via the engine so the gradient tape records the op. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the SELU function for a single input value. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the SELU activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The alpha parameter for the SELU function, which controls the negative saturation value. |
| `_lambda` | The scaling factor for the SELU function. |

