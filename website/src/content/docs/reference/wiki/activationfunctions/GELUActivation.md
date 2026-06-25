---
title: "GELUActivation<T>"
description: "Implements the Gaussian Error Linear Unit (GELU) activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Gaussian Error Linear Unit (GELU) activation function for neural networks.

## For Beginners

The GELU activation function is a modern activation function used in many 
state-of-the-art neural networks, including transformers like BERT and GPT.

Think of GELU as a "smoother" version of ReLU that:

- Keeps most positive values (like ReLU does)
- Gradually reduces small positive values
- Gradually allows some small negative values through
- Blocks large negative values (like ReLU does)

GELU can be thought of as multiplying the input by the probability that the input is positive.
This creates a smooth curve that transitions naturally between allowing and blocking values,
rather than having a sharp cutoff like ReLU.

GELU is widely used in modern language models and has been shown to perform better than
older activation functions in many deep learning tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether GELU supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the GELU activation function to a single input value. |
| `Activate(Tensor<>)` | Applies the GELU activation function to each element in a tensor. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the GELU function for a single input value. |
| `Derivative(Tensor<>)` | Calculates the derivative of the GELU function for each element in a tensor. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the GELU activation function on GPU. |
| `SupportsScalarOperations` | Indicates that this activation function supports operations on individual scalar values. |

