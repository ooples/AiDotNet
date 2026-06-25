---
title: "GaussianActivation<T>"
description: "Implements the Gaussian activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Gaussian activation function for neural networks.

## For Beginners

The Gaussian activation function is based on the bell-shaped curve that you might 
recognize from statistics (the "normal distribution" or "bell curve"). 

Key properties of the Gaussian activation function:

- It outputs values between 0 and 1
- The highest output (1) occurs when the input is 0
- As inputs move away from 0 (either positive or negative), the output approaches 0
- It's symmetric around the y-axis (f(-x) = f(x))

Unlike many other activation functions, Gaussian responds strongly to inputs near zero and 
weakly to inputs far from zero in either direction. This makes it useful for:

- Radial Basis Function (RBF) networks
- Pattern recognition tasks
- Problems where distance from a central point is important

The mathematical formula is: f(x) = exp(-x²)

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Gaussian activation function to a single input value. |
| `Activate(Tensor<>)` | Applies Gaussian to a tensor via engine primitives so the gradient tape records every step of `exp(-x²)`. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the Gaussian function for a single input value. |
| `SupportsScalarOperations` | Indicates that this activation function supports operations on individual scalar values. |

