---
title: "LogSoftminActivation<T>"
description: "Implements the LogSoftmin activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the LogSoftmin activation function for neural networks.

## For Beginners

The LogSoftmin function is similar to LogSoftmax, but focuses on the smallest values 
instead of the largest values in a vector.

It works in two steps:

1. First, it applies the "softmin" function, which gives more weight to smaller numbers in the input

(the opposite of softmax, which emphasizes larger numbers).

2. Then, it takes the natural logarithm of these values.

While LogSoftmax is used to highlight the largest values (useful for finding the most likely class in 
classification), LogSoftmin can be useful when you want to focus on the smallest values in your data.

Like LogSoftmax, this function operates on vectors (collections of numbers) rather than individual values,
because it needs to consider all values together to determine their relative importance.

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies LogSoftmin to a tensor via engine primitives so the gradient tape records every step. |
| `Activate(Vector<>)` | Applies the LogSoftmin activation function to a vector of values. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative(Vector<>)` | Calculates the derivative (Jacobian matrix) of the LogSoftmin function for a vector input. |
| `SupportsScalarOperations` | Determines if the activation function supports operations on individual scalar values. |

