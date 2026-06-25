---
title: "ReLU6Activation<T>"
description: "Implements the ReLU6 (Rectified Linear Unit capped at 6) activation function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the ReLU6 (Rectified Linear Unit capped at 6) activation function.

## For Beginners

ReLU6 works like regular ReLU but adds an upper limit of 6.

While ReLU allows any positive value to pass through unchanged, ReLU6 caps the output at 6:

- Negative inputs become 0 (same as ReLU)
- Values between 0 and 6 pass through unchanged
- Values above 6 are capped at 6

This is particularly useful in mobile neural networks (like MobileNet) because:

- It prevents activations from becoming too large, improving numerical stability
- It works well with low-precision arithmetic (like 8-bit integers) on mobile devices
- It helps the network learn more robust features by limiting extreme activations

## How It Works

ReLU6 is a variant of the ReLU activation function that clips the output at 6.
It is defined as: f(x) = min(max(0, x), 6)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReLU6Activation` | Initializes a new instance of the `ReLU6Activation` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the ReLU6 activation function to a single input value. |
| `Activate(Tensor<>)` | Applies the ReLU6 activation function to each element in a tensor via Engine ops so the gradient tape records the full chain. |
| `Activate(Vector<>)` | Applies the ReLU6 activation function to each element in a vector. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the ReLU6 function for a single input value. |
| `Derivative(Tensor<>)` | Calculates the derivative of the ReLU6 function for each element in a tensor. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of the ReLU6 function for a vector input. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

