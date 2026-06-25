---
title: "SiLUActivation<T>"
description: "Implements the SiLU (Sigmoid Linear Unit) activation function, also known as Swish."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the SiLU (Sigmoid Linear Unit) activation function, also known as Swish.

## For Beginners

SiLU (or Swish) is a relatively new activation function that has become 
popular in modern neural networks. Unlike simpler functions like ReLU that either pass 
a value through or block it, SiLU smoothly scales inputs based on their value. It keeps 
most positive values, reduces small positive values, and allows some negative values to 
pass through (but reduced in magnitude). This smooth behavior helps neural networks learn 
more complex patterns. SiLU is used in many state-of-the-art models, especially in deep 
learning applications like computer vision and natural language processing.

## How It Works

The SiLU function is defined as f(x) = x * sigmoid(x), where sigmoid(x) = 1/(1+e^(-x)).
It was introduced in 2017 and has shown strong performance in deep neural networks.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether SiLU supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the SiLU activation function to a single input value. |
| `Activate(Tensor<>)` | Applies SiLU/Swish to a tensor via the engine so the gradient tape records it. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the SiLU function for a single input value. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the SiLU activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

