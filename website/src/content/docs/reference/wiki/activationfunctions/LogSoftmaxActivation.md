---
title: "LogSoftmaxActivation<T>"
description: "Implements the LogSoftmax activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the LogSoftmax activation function for neural networks.

## For Beginners

The LogSoftmax function combines two operations:

1. First, it applies the "softmax" function, which converts a vector of numbers into probabilities

(values between 0 and 1 that sum to 1).

2. Then, it takes the natural logarithm of these probabilities.

This function is commonly used in the final layer of neural networks for classification problems,
especially when combined with Negative Log-Likelihood loss. It helps with:

- Numerical stability (preventing extremely small or large numbers)
- Making the math work better during training
- Producing outputs that work well for calculating classification probabilities

Unlike most activation functions, LogSoftmax operates on vectors (collections of numbers) rather than
individual values, because it needs to consider all outputs together to calculate probabilities.

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies LogSoftmax to a tensor via `Engine.TensorLogSoftmax` so the gradient tape records the op. |
| `Activate(Vector<>)` | Applies the LogSoftmax activation function to a vector of values. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative(Vector<>)` | Calculates the derivative (Jacobian matrix) of the LogSoftmax function for a vector input. |
| `SupportsScalarOperations` | Determines if the activation function supports operations on individual scalar values. |

