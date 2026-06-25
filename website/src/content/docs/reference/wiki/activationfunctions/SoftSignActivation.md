---
title: "SoftSignActivation<T>"
description: "Implements the SoftSign activation function, which is a smooth alternative to the tanh function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the SoftSign activation function, which is a smooth alternative to the tanh function.

## For Beginners

SoftSign is an activation function that squeezes input values into a range between -1 and 1.
It's similar to the tanh function but approaches its limits more slowly.

Key properties of SoftSign:

- For input of 0, the output is 0
- For large positive inputs, the output approaches 1 (but never reaches it)
- For large negative inputs, the output approaches -1 (but never reaches it)
- The function is smooth everywhere, making it easier to train
- Unlike tanh, SoftSign has "polynomial" tails, meaning it approaches its limits more gradually

This gradual approach to limits can help prevent neurons from becoming "saturated" (stuck at extreme values)
during training, which can be an advantage in some neural network architectures.

## How It Works

The SoftSign function is defined as: f(x) = x / (1 + |x|), where |x| is the absolute value of x.
It maps any input value to an output between -1 and 1, similar to tanh but with different properties.

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the SoftSign activation function to a single input value. |
| `Activate(Tensor<>)` | Applies SoftSign to a tensor via engine primitives so the gradient tape records every step of `x / (1 + \|x\|)`. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the SoftSign function for a given input. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

