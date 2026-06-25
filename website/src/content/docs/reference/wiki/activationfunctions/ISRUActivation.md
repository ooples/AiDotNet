---
title: "ISRUActivation<T>"
description: "Implements the Inverse Square Root Unit (ISRU) activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Inverse Square Root Unit (ISRU) activation function for neural networks.

## For Beginners

The ISRU (Inverse Square Root Unit) activation function is designed to:

- Keep the output values bounded (they never grow too large)
- Preserve the sign of the input (positive inputs give positive outputs, negative inputs give negative outputs)
- Allow gradients to flow more easily during training compared to some other functions

The function looks like an "S" shape that's been stretched horizontally, similar to tanh but with 
different mathematical properties. It approaches +1 for large positive inputs and -1 for large negative inputs,
but never quite reaches these values.

The a (alpha) parameter controls how quickly the function "saturates" (flattens out):

- Smaller a values make the function change more gradually
- Larger a values make the function change more abruptly

ISRU is useful in neural networks where you want bounded outputs but need to avoid the vanishing 
gradient problem that affects some other activation functions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ISRUActivation(Double)` | Initializes a new instance of the ISRU activation function with the specified alpha parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the ISRU activation function to a single value. |
| `Activate(Tensor<>)` | Applies ISRU to a tensor via engine primitives so the gradient tape records every step of `x / sqrt(1 + αx²)`. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative (gradient) of the ISRU function for a single value. |
| `SupportsScalarOperations` | Indicates whether this activation function can operate on individual scalar values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The alpha parameter that controls the shape of the ISRU function. |

