---
title: "RReLUActivation<T>"
description: "Implements the Randomized Rectified Linear Unit (RReLU) activation function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Randomized Rectified Linear Unit (RReLU) activation function.

## For Beginners

RReLU (Randomized Rectified Linear Unit) is like a "smart filter" for neural networks.
When data is positive, it lets it pass through unchanged. When data is negative, instead of setting it to zero
(like regular ReLU), it reduces the value by multiplying it by a small random number. This randomness helps
prevent "dead neurons" (neurons that stop learning) and can improve the network's ability to learn.
During training, this random factor changes; during testing/inference, a fixed average value is used.

## How It Works

RReLU is a variation of the ReLU activation function that introduces randomness during training.
For positive inputs, it behaves like standard ReLU (returning the input unchanged).
For negative inputs, it multiplies the input by a random factor (alpha) between the specified lower and upper bounds.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RReLUActivation(Double,Double)` | Initializes a new instance of the RReLU activation function with specified bounds for the random alpha parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the RReLU activation function to a single input value. |
| `Activate(Tensor<>)` | Applies RReLU to a tensor via engine primitives so the gradient tape records every step. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the RReLU function for a single input value. |
| `SetTrainingMode(Boolean)` | Sets the activation function to either training or inference mode. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The current alpha value used to scale negative inputs. |
| `_isTraining` | Indicates whether the activation function is in training mode (true) or inference mode (false). |
| `_lowerBound` | The minimum value for the random alpha parameter. |
| `_random` | Random number generator used to create the alpha value during training. |
| `_upperBound` | The maximum value for the random alpha parameter. |

