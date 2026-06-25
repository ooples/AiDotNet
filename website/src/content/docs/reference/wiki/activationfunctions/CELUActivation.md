---
title: "CELUActivation<T>"
description: "Implements the Continuously Differentiable Exponential Linear Unit (CELU) activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Continuously Differentiable Exponential Linear Unit (CELU) activation function for neural networks.

## For Beginners

The CELU activation function is an improved version of the popular ReLU function.
While ReLU simply turns negative values to zero (which can cause "dead neurons"), CELU replaces
negative values with a smooth exponential curve that approaches a negative limit.

Key benefits of CELU:

- For positive inputs, it behaves exactly like ReLU (returns the input value)
- For negative inputs, it returns a negative value that smoothly approaches -a
- This smooth transition helps prevent "dead neurons" during training
- The a parameter controls how quickly the function approaches its negative limit

CELU is particularly useful in deep neural networks where maintaining gradient flow
through all neurons is important for effective learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CELUActivation(Double)` | Initializes a new instance of the CELUActivation class with the specified alpha parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the CELU activation function to a single input value. |
| `Activate(Tensor<>)` | Applies CELU to a tensor via engine primitives so the gradient tape records every step. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the CELU function for a single input value. |
| `SupportsScalarOperations` | Indicates that this activation function supports operations on individual scalar values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The alpha parameter that controls the negative saturation value of the function. |

