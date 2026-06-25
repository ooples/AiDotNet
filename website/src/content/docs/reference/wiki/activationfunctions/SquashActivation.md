---
title: "SquashActivation<T>"
description: "Implements the Squash activation function, which normalizes vectors to have a magnitude between 0 and 1."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Squash activation function, which normalizes vectors to have a magnitude between 0 and 1.

## For Beginners

The Squash function is different from most activation functions because it works on
entire vectors (groups of numbers) rather than individual numbers. Think of a vector as an arrow
pointing in some direction with a certain length.

What Squash does:

1. It keeps the arrow pointing in the same direction
2. It adjusts the length of the arrow to be between 0 and 1
3. Short vectors remain almost the same length
4. Long vectors get "squashed" to be closer to length 1

This is useful in advanced neural networks like capsule networks, where the direction of a vector
represents important features, and the length represents the probability that those features exist.

## How It Works

The Squash activation function maps input vectors to have a magnitude (length) between 0 and 1 while
preserving their direction. This is particularly useful in capsule networks and other neural network
architectures where vector orientation is important.

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | This method is not supported by SquashActivation as it only operates on vectors. |
| `Activate(Tensor<>)` | Applies the Squash activation function to a batch of vectors stored in a tensor. |
| `Activate(Vector<>)` | Applies the Squash activation function to a vector input. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | This method is not supported by SquashActivation as it only operates on vectors. |
| `Derivative(Tensor<>)` | Calculates the Jacobian matrices for a batch of vectors stored in a tensor. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of the Squash function at the given input vector. |
| `SupportsScalarOperations` | Indicates whether this activation function supports operations on individual scalar values. |

