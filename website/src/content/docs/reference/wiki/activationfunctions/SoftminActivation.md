---
title: "SoftminActivation<T>"
description: "Implements the Softmin activation function, which is the opposite of Softmax and highlights the smallest values in a vector."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Softmin activation function, which is the opposite of Softmax and highlights the smallest values in a vector.

## For Beginners

While Softmax emphasizes the largest values in a vector, Softmin does the opposite - it gives
more weight to smaller values. Think of it as a "smooth minimum" function. For example, if you have scores
[5, 2, 8], Softmax would highlight 8 (the maximum), but Softmin would highlight 2 (the minimum).

Softmin is less commonly used than Softmax but can be useful in scenarios where you want to identify or
emphasize the smallest values in a set, such as finding the closest points in certain distance-based algorithms
or when you want to assign higher probabilities to smaller values.

Like Softmax, Softmin outputs values between 0 and 1 that sum to 1, creating a probability distribution.

## How It Works

The Softmin function takes a vector of real numbers and transforms it into a probability distribution
that emphasizes smaller values. It's defined as: softmin(x_i) = exp(-x_i) / sum(exp(-x_j)) for all j in the vector.

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies Softmin to a tensor via engine primitives so the gradient tape records every step. |
| `Activate(Vector<>)` | Applies the Softmin activation function to a vector of input values. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of the Softmin function for a vector input. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

