---
title: "TrainableParameterAttribute"
description: "Marks a `Tensor` field as a trainable parameter that should be registered with the gradient tape training system."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Marks a `Tensor` field as a trainable parameter
that should be registered with the gradient tape training system.

## For Beginners

Put this attribute on any tensor field that the network should
learn during training. The framework handles everything else — registering it, exposing it
to the optimizer, and clearing gradients between training steps.

## How It Works

The `TrainableParameterGenerator` source generator discovers fields marked with this
attribute and automatically emits:

- `GetTrainableParameters()` — returns all marked fields in declaration order
- `SetTrainableParameters(Tensor<T>[])` — updates each marked field from the array
- `ZeroGrad()` — zeros gradient fields discovered by convention ({fieldName}Gradient)

This is the equivalent of PyTorch's `nn.Parameter` — marking a tensor as trainable
makes it automatically visible to the optimizer and gradient tape with zero manual boilerplate.

**Convention for gradient fields:** For a parameter field named `_weights`,
the generator looks for `_weightsGradient` (nullable `Tensor<T>?`).
If found, `ZeroGrad()` will zero or null it automatically.

## Properties

| Property | Summary |
|:-----|:--------|
| `Optional` | Gets or sets whether this parameter is *optional*: a conditionally-used, lazily-materialized field that stays a zero-length `[0,0]` placeholder until (and unless) the layer actually needs it. |
| `Order` | Gets or sets the display order of this parameter in `GetTrainableParameters()`. |
| `Role` | Gets or sets the role of this parameter for GPU memory management hints. |

