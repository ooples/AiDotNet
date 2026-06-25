---
title: "ScaledTanhActivation<T>"
description: "Implements the Scaled Hyperbolic Tangent (tanh) activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Scaled Hyperbolic Tangent (tanh) activation function for neural networks.

## For Beginners

The Scaled Tanh activation function is a parameterized version of the standard
hyperbolic tangent function. Like the standard tanh, it outputs values between -1 and 1, making
it useful for neural networks where you want the output to be centered around zero.

The mathematical formula is: f(x) = (1 - e^(-ßx)) / (1 + e^(-ßx))

This is equivalent to the standard tanh function when ß = 2, and has these key properties:

- Outputs values between -1 and 1
- Is symmetric around the origin (f(-x) = -f(x))
- The parameter ß (beta) controls the steepness of the curve
- When ß = 2, this is exactly equivalent to the standard tanh function

When to use it:

- When you need outputs centered around zero
- For hidden layers in many types of neural networks
- When dealing with data that naturally has both positive and negative values
- When you want to control the steepness of the activation function

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ScaledTanhActivation(Double,Double)` | Initializes a new instance of the ScaledTanhActivation class with the specified steepness parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Scaled Tanh activation function to a single input value. |
| `Activate(Tensor<>)` | Applies ScaledTanh to a tensor via engine primitives so the gradient tape records every step. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the Scaled Tanh function for a single input value. |
| `SupportsScalarOperations` | Indicates that this activation function supports operations on individual scalar values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_beta` | The steepness parameter that controls how quickly the function transitions between -1 and 1. |

