---
title: "TaylorSoftmaxActivation<T>"
description: "Implements the Taylor Softmax activation function, which is a computationally efficient approximation of the standard Softmax function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Taylor Softmax activation function, which is a computationally efficient approximation of the standard Softmax function.

## For Beginners

The Taylor Softmax function is a variation of the standard Softmax function that uses a mathematical
technique called "Taylor series expansion" to approximate the exponential function. This makes it computationally
more efficient while maintaining similar behavior to the standard Softmax.

Softmax functions are commonly used in the output layer of neural networks for multi-class classification problems.
They convert a vector of numbers into a probability distribution (all values are positive and sum to 1).

For example, if you have three output neurons with values [2.0, 1.0, 0.5], the Softmax function will convert
these to probabilities like [0.6, 0.25, 0.15], which sum to 1.0. This makes it easy to interpret the outputs
as probabilities for each class.

The "Taylor" part refers to using a mathematical approximation (Taylor series) instead of calculating the
full exponential function, which can be faster but slightly less accurate.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TaylorSoftmaxActivation(Int32)` | Initializes a new instance of the TaylorSoftmaxActivation class with the specified order of approximation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies TaylorSoftmax to a tensor via engine primitives so the gradient tape records every step. |
| `Activate(Vector<>)` | Applies the Taylor Softmax activation function to a vector of input values. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of partial derivatives for the Taylor Softmax function. |
| `SupportsScalarOperations` | Indicates whether this activation function supports operations on individual scalar values. |
| `TaylorExp(,Int32)` | Approximates the exponential function e^x using a Taylor series expansion. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_order` | The order of the Taylor series approximation used for the exponential function. |

