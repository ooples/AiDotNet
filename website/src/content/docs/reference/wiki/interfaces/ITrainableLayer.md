---
title: "ITrainableLayer<T>"
description: "Defines a neural network layer with trainable parameters that can be used with tape-based automatic differentiation (autodiff)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a neural network layer with trainable parameters that can be used with
tape-based automatic differentiation (autodiff).

## For Beginners

When training a neural network, we need two things:

- Access to the layer's learnable values (weights and biases) so the optimizer

can update them after computing gradients.

- A way to clear old gradient information before each training step, so gradients

from different batches don't accumulate incorrectly.

This interface provides both capabilities, enabling the PyTorch-style training loop:

## How It Works

This interface bridges layers with `GradientTape<T>`
by exposing the exact `Tensor` instances that the layer uses during
its forward pass. The gradient tape tracks operations by tensor reference identity,
so the tensors returned here must be the same objects passed to engine operations
in `Tensor{`.

**PyTorch equivalent:** This combines the role of `nn.Module.parameters()`
(which yields the same Parameter objects used in forward) and
`optimizer.zero_grad()` (which clears .grad on each parameter).

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTrainableParameters` | Returns the trainable parameter tensors used by this layer during forward execution. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces this layer's trainable parameter tensors with the provided tensors. |
| `ZeroGrad` | Clears all accumulated gradients on this layer's trainable parameters. |

