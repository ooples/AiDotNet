---
title: "PaddingLayer<T>"
description: "Represents a layer that adds padding to the input tensor."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that adds padding to the input tensor.

## For Beginners

This layer adds extra space around the edges of your data.

Think of it like adding a frame around a picture:

- You have an image (your input data)
- The padding adds extra space around all sides of the image
- The padding is filled with zeros by default

This is useful for:

- Preserving the size of images when applying convolutions
- Preventing loss of information at the edges of the data
- Giving convolutional filters more context at the boundaries

For example, if you have a 28×28 image and add padding of 2 pixels on all sides,
you get a 32×32 image with your original data in the center and zeros around the edges.

## How It Works

The PaddingLayer adds a specified amount of padding around the edges of the input tensor.
This is commonly used in convolutional neural networks to preserve spatial dimensions
after convolution operations or to provide additional context at the boundaries of the input.
The padding is added symmetrically on both sides of each dimension of the input tensor.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PaddingLayer(Int32[],IActivationFunction<>)` | Initializes a new instance of the `PaddingLayer` class with the specified input shape, padding, and a scalar activation function. |
| `PaddingLayer(Int32[],IVectorActivationFunction<>)` | Initializes a new instance of the `PaddingLayer` class with the specified input shape, padding, and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` |  |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Int32[])` | Calculates the output shape of the padding layer based on the input shape and padding amounts. |
| `Forward(Tensor<>)` | Performs the forward pass of the padding layer. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetParameters` | Gets all trainable parameters from the padding layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves output shape on first forward by adding 2*padding to input.Shape per axis. |
| `ResetState` | Resets the internal state of the padding layer. |
| `UpdateParameters()` | Updates the parameters of the padding layer using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuCachedInputShape` | Cached GPU input shape for backward pass. |
| `_lastInput` | The input tensor from the most recent forward pass. |
| `_padding` | The amount of padding to add to each dimension of the input tensor. |

