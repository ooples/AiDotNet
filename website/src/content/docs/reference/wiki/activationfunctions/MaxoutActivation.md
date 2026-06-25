---
title: "MaxoutActivation<T>"
description: "Implements the Maxout activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Maxout activation function for neural networks.

## For Beginners

The Maxout activation function is different from most activation functions.
Instead of applying a mathematical formula to each value, it:

1. Groups your input values into small sets (e.g., groups of 2, 3, or more values)
2. For each group, it selects only the largest (maximum) value
3. The output is smaller than the input (it's reduced by a factor equal to the group size)

For example, with groups of 2 (numPieces = 2):
Input: [1, 5, 3, 7]
Groups: [1, 5] and [3, 7]
Output: [5, 7] (the maximum value from each group)

Maxout is powerful because it can learn to approximate many different activation functions,
making it very flexible. However, it requires more parameters in your neural network.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaxoutActivation(Int32)` | Creates a new instance of the Maxout activation function. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies Maxout to a tensor via engine primitives so the gradient tape records every step. |
| `Activate(Vector<>)` | Applies the Maxout activation function to a vector of values. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative(Vector<>)` | Calculates the derivative (Jacobian matrix) of the Maxout function for a vector input. |
| `SupportsScalarOperations` | Determines if the activation function supports operations on individual scalar values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numPieces` | The number of pieces (group size) for the Maxout function. |

