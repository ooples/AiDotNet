---
title: "LiSHTActivation<T>"
description: "Implements the Linearly Scaled Hyperbolic Tangent (LiSHT) activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Linearly Scaled Hyperbolic Tangent (LiSHT) activation function for neural networks.

## For Beginners

The LiSHT activation function combines the input value with its hyperbolic tangent.

The formula is: f(x) = x * tanh(x)

This means:

- For positive inputs: The output is positive but scaled down
- For negative inputs: The output is negative but scaled down
- For zero: The output is zero

LiSHT has several advantages:

- It doesn't suffer from the "vanishing gradient problem" (where learning becomes very slow)
- It's smooth everywhere (unlike ReLU which has a sharp corner at zero)
- It naturally keeps values in a reasonable range

Think of it as a function that "squeezes" large values while preserving the sign and 
allowing small values to pass through with minimal change.

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the LiSHT activation function to a single input value. |
| `Activate(Tensor<>)` | Applies LiSHT to a tensor via engine primitives so the gradient tape records every step of `x * tanh(x)`. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the LiSHT function for a single input value. |
| `SupportsScalarOperations` | Determines if the activation function supports operations on individual scalar values. |

