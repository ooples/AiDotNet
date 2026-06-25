---
title: "PReLULayer<T>"
description: "Implements a Parametric ReLU (PReLU) layer with learnable negative-slope coefficients."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a Parametric ReLU (PReLU) layer with learnable negative-slope coefficients.

## For Beginners

PReLU is like Leaky ReLU except the "leakiness" is learned from data instead
of being set by you. For positive inputs the output is unchanged; for negative inputs the output
is the input scaled by α. The network adjusts α during training to control how much signal to let
through on the negative side. Per-channel α lets different channels learn different leaks, which
often works better than a single shared α for convolutional networks.

## How It Works

PReLU is a learnable variant of Leaky ReLU where the negative-slope coefficient `α` is
a trainable parameter rather than a fixed hyperparameter. The function is:

`f(x) = max(0, x) + α * min(0, x) = ReLU(x) - α * ReLU(-x)`

Two parameterization modes are supported, matching PyTorch's `nn.PReLU`:

Introduced in He et al., "Delving Deep into Rectifiers" (2015), which also introduced the Kaiming
initialization. The paper's recommended initial α is 0.25, which is this layer's default.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PReLULayer(Int32,Int32,Double)` | Initializes a new `PReLULayer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters (the length of α). |
| `SupportsTraining` | Gets a value indicating that this layer has trainable parameters (α). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `Forward(Tensor<>)` | Performs the forward pass: `ReLU(x) - α · ReLU(-x)`, all ops on the gradient tape. |
| `GetAlphaTensor` | Gets the current α tensor. |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` | Resolves broadcast shape and validates channel-count compatibility on first forward. |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `TryDeclareShape` | AiDotNet#1370 shape oracle override: PReLU's α weight tensor is fully determined by the constructor argument `numParameters` and is allocated + registered as a trainable parameter at construction time (line 98–103 above). |
| `UpdateParameters()` | Legacy scalar-learning-rate parameter update. |

