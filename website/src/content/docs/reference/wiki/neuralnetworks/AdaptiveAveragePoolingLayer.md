---
title: "AdaptiveAveragePoolingLayer<T>"
description: "Implements adaptive average pooling that outputs a fixed spatial size regardless of input dimensions."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements adaptive average pooling that outputs a fixed spatial size regardless of input dimensions.

## For Beginners

Regular pooling uses a fixed window size (like 2x2) and reduces the image.
Adaptive pooling works in reverse: you specify the output size you want (like 1x1), and it
automatically figures out how to pool the entire input to get that size.

For example:

- Input: 14x14, Output: 1x1 → Pools each entire channel to a single value
- Input: 7x7, Output: 1x1 → Same result: each channel becomes one value
- Input: 56x56, Output: 7x7 → Divides into 7x7 regions and averages each

This is commonly used in ResNet and other architectures for "global average pooling" where
the final feature maps are reduced to a single value per channel before classification.

## How It Works

Adaptive average pooling automatically calculates the required kernel size and stride to produce
an output of the specified dimensions. This is particularly useful when you want to handle
variable input sizes but need a fixed output size (e.g., before a fully connected layer).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaptiveAveragePoolingLayer(Int32,Int32)` | Initializes a new instance of the `AdaptiveAveragePoolingLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of adaptive average pooling. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass of adaptive average pooling on GPU tensors. |
| `GetParameters` | Gets all trainable parameters. |
| `GlobalPool` | Creates a global average pooling layer that pools to 1x1. |
| `OnFirstForward(Tensor<>)` | Resolves channels and input spatial dims on first forward. |
| `ResetState` | Resets the internal state. |
| `UpdateParameters()` | Updates the parameters. |

