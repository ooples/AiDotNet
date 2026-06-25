---
title: "BentIdentityActivation<T>"
description: "Implements the Bent Identity activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Bent Identity activation function for neural networks.

## For Beginners

The Bent Identity activation function is a smoother alternative to the ReLU function.
It behaves similarly to a linear function for positive inputs but has a gentle curve for negative inputs.
This helps prevent the "dying neuron" problem that can occur with ReLU, where neurons can get stuck
outputting zero.

The mathematical formula is: f(x) = ((sqrt(x² + 1) - 1) / 2) + x

Key properties:

- Always produces a non-zero gradient, helping with training
- Approximates linear behavior for large positive values
- Provides a smooth transition around zero
- Has no upper or lower bounds (unlike sigmoid or tanh)

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Bent Identity activation function to a single input value. |
| `Activate(Tensor<>)` | Applies BentIdentity to a tensor via engine primitives so the gradient tape records every step of `(sqrt(x² + 1) - 1) / 2 + x`. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the Bent Identity function for a single input value. |
| `SupportsScalarOperations` | Indicates that this activation function supports operations on individual scalar values. |

