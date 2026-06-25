---
title: "GumbelSoftmaxActivation<T>"
description: "Implements the Gumbel-Softmax activation function for neural networks, which enables differentiable sampling from discrete distributions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Gumbel-Softmax activation function for neural networks, which enables
differentiable sampling from discrete distributions.

## For Beginners

The Gumbel-Softmax is a special activation function that helps neural networks
make categorical (multiple-choice) decisions while still allowing for gradient-based learning.

Imagine you want your neural network to choose between several options (like choosing a word
from a vocabulary). Normally, this would require a non-differentiable "hard" selection, which
makes training difficult. Gumbel-Softmax solves this by:

1. Adding randomness (Gumbel noise) to the inputs
2. Using a "temperature" parameter to control how "certain" the choices are
3. Producing a probability distribution that can approximate discrete choices

At high temperatures, the output is very "soft" (all options get some probability).
At low temperatures, the output becomes more like a one-hot vector (one option gets almost all probability).

This technique is widely used in generative models, reinforcement learning, and any neural network
that needs to make discrete choices while remaining differentiable.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GumbelSoftmaxActivation(Double,Nullable<Int32>)` | Initializes a new instance of the GumbelSoftmaxActivation class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies Gumbel-Softmax to a tensor via engine primitives so the gradient tape records every step. |
| `Activate(Vector<>)` | Applies the Gumbel-Softmax activation function to a vector of input values. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative(Vector<>)` | Calculates the derivative (Jacobian matrix) of the Gumbel-Softmax function for a vector of input values. |
| `SampleGumbel(Int32)` | Generates a vector of random values from the Gumbel distribution. |
| `Softmax(Vector<>)` | Applies the softmax function to a vector of logits, using the current temperature. |
| `SupportsScalarOperations` | Indicates whether this activation function supports operations on individual scalar values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_random` | Random number generator used for sampling Gumbel noise. |
| `_temperature` | Controls the "sharpness" of the output distribution. |

