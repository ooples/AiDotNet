---
title: "SoftmaxActivation<T>"
description: "Implements the Softmax activation function, which converts a vector of real numbers into a probability distribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Softmax activation function, which converts a vector of real numbers into a probability distribution.

## For Beginners

Softmax is commonly used in the output layer of neural networks for classification problems.
Think of it as a way to convert raw scores (called "logits") into probabilities. For example, if you're
classifying images into 3 categories (cat, dog, bird), the neural network might output raw scores like
[2.5, 1.2, 0.8]. Softmax converts these to probabilities like [0.65, 0.22, 0.13], which sum to 1.0 (or 100%).
This makes it easy to interpret the highest value as the model's prediction (in this case, "cat" with 65% confidence).

Unlike other activation functions that work on single values, Softmax needs to see all values at once because
it normalizes them relative to each other.

## How It Works

The Softmax function takes a vector of real numbers and normalizes it into a probability distribution,
where each value is between 0 and 1, and all values sum to 1. It's defined as:
softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j in the vector.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuTraining` | Gets whether Softmax supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies the Softmax activation function to each element in a tensor. |
| `Activate(Vector<>)` | Applies the Softmax activation function to a vector of input values. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of the Softmax function for a vector input. |
| `ForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the Softmax activation function on GPU. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

