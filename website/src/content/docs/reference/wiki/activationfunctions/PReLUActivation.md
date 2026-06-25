---
title: "PReLUActivation<T>"
description: "Implements the Parametric Rectified Linear Unit (PReLU) activation function for neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Parametric Rectified Linear Unit (PReLU) activation function for neural networks.

## For Beginners

PReLU is an improved version of the popular ReLU activation function.

While ReLU completely blocks negative values (turning them to zero), PReLU allows
a small portion of negative values to pass through, controlled by a parameter called "alpha".

How PReLU works:

- For positive inputs (x > 0): PReLU returns the input unchanged (just like ReLU)
- For negative inputs (x = 0): PReLU returns alpha * x (a scaled-down version of the input)

The alpha parameter is typically a small positive number (default 0.01). This "leakiness"
helps prevent a problem called "dying ReLU" where neurons can get stuck and stop learning.

PReLU can be thought of as a sloped line for negative inputs rather than a flat line at zero.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PReLUActivation(Double)` | Initializes a new instance of the PReLU activation function with the specified alpha value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the PReLU activation function to a single input value. |
| `Activate(Tensor<>)` | Applies PReLU to a tensor via engine primitives so the gradient tape records every step. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative (gradient) of the PReLU function for a single input value. |
| `SupportsScalarOperations` | Determines if the activation function supports operations on individual scalar values. |
| `UpdateAlpha()` | Updates the alpha parameter of the PReLU function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The scaling factor applied to negative inputs. |

