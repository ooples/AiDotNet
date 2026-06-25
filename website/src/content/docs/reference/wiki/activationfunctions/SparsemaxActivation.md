---
title: "SparsemaxActivation<T>"
description: "Implements the Sparsemax activation function, which is an alternative to Softmax that can produce sparse probability distributions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Sparsemax activation function, which is an alternative to Softmax that can produce sparse probability distributions.

## For Beginners

Sparsemax is an advanced activation function used primarily in the output layer of 
neural networks for classification tasks. While Softmax always gives some probability to every possible 
class (even if very small), Sparsemax can assign exactly zero probability to unlikely classes.

Think of it like this:

- Softmax is like saying "I'm 80% sure it's a dog, 15% sure it's a cat, and 5% sure it's something else"
- Sparsemax might say "I'm 90% sure it's a dog, 10% sure it's a cat, and 0% sure it's anything else"

This "sparsity" (having many zeros) can be useful when you have many possible classes but only a few
are likely to be correct. It makes the model's predictions more focused and interpretable.

## How It Works

Sparsemax maps input vectors to probability distributions, similar to Softmax, but with the key difference
that Sparsemax can assign exactly zero probability to low-scoring classes, creating sparse outputs.

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies Sparsemax to a tensor via `Engine.Sparsemax` so the gradient tape records the op. |
| `Activate(Vector<>)` | Applies the Sparsemax activation function to a vector of inputs. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of the Sparsemax function for a given input vector. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

