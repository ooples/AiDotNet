---
title: "ThresholdedReLUActivation<T>"
description: "Implements the Thresholded ReLU activation function, a variant of the standard ReLU function with an adjustable threshold."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Thresholded ReLU activation function, a variant of the standard ReLU function with an adjustable threshold.

## For Beginners

The Thresholded ReLU (Rectified Linear Unit) is a variation of the standard ReLU activation function.

While a standard ReLU outputs the input value when it's positive and zero when it's negative (f(x) = max(0, x)),
the Thresholded ReLU adds an additional parameter called "theta" (?) that acts as a threshold.

The Thresholded ReLU only activates (returns the input value) when the input exceeds this threshold.
Otherwise, it returns zero. The formula is:

f(x) = x if x > ?, otherwise f(x) = 0

This allows the neural network to ignore small positive activations that might be noise, potentially
creating more robust models. By adjusting the threshold value, you can control how sensitive the
activation function is to input signals.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThresholdedReLUActivation(Double)` | Initializes a new instance of the ThresholdedReLUActivation class with the specified threshold value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Thresholded ReLU activation function to an input value. |
| `Activate(Tensor<>)` | Applies ThresholdedReLU to a tensor via engine primitives so the gradient tape records every step. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the Thresholded ReLU function for a given input. |
| `SupportsScalarOperations` | Indicates whether this activation function supports operations on individual scalar values. |
| `UpdateTheta()` | Updates the threshold value used by the activation function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_theta` | The threshold value that determines when the activation function returns the input value versus zero. |

