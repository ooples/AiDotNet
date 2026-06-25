---
title: "MCDropoutLayer<T>"
description: "Implements Monte Carlo Dropout layer for uncertainty estimation in neural networks."
section: "API Reference"
---

`Layers` · `AiDotNet.UncertaintyQuantification.Layers`

Implements Monte Carlo Dropout layer for uncertainty estimation in neural networks.

## For Beginners

Monte Carlo Dropout is a simple yet powerful technique for estimating uncertainty.

Unlike regular dropout which is only active during training, MC Dropout keeps dropout active
during prediction as well. By running multiple predictions with different random dropout masks,
we get a distribution of predictions. The spread of this distribution tells us how uncertain
the model is.

Think of it like asking multiple slightly different versions of the same expert for their opinion.
If they all agree, you can be confident. If they disagree widely, there's high uncertainty.

This is particularly useful for:

- Detecting out-of-distribution samples
- Active learning (selecting which data to label next)
- Safety-critical applications (knowing when to defer to a human expert)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MCDropoutLayer(Double,Boolean,Nullable<Int32>)` | Initializes a new instance of the MCDropoutLayer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MonteCarloMode` | Gets or sets whether Monte Carlo mode is enabled (applies dropout during inference). |
| `SupportsTraining` | Gets a value indicating whether this layer supports training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `Forward(Tensor<>)` | Performs the forward pass of the MC dropout layer. |
| `GetParameters` | Gets the trainable parameters (empty for dropout layers). |
| `ResetState` | Resets the internal state of the layer. |
| `SetParameters(Vector<>)` | Sets the trainable parameters (no-op for dropout layers). |
| `UpdateParameters()` | Updates the parameters (no-op for dropout layers). |

