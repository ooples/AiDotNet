---
title: "UpsamplingLayer<T>"
description: "Represents an upsampling layer that increases the spatial dimensions of input tensors using nearest-neighbor interpolation."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents an upsampling layer that increases the spatial dimensions of input tensors using nearest-neighbor interpolation.

## For Beginners

This layer makes images or feature maps larger by simply repeating pixels.

Think of it like zooming in on a digital image:

- When you zoom in on a pixelated image, each original pixel becomes a larger square
- This layer does the same thing to feature maps inside the neural network
- It's like stretching an image without adding any new information

For example, with a scale factor of 2:

- A 4Ã—4 image becomes an 8Ã—8 image
- Each pixel in the original image is copied to a 2Ã—2 block in the output
- This creates a larger image that preserves the original content but with more pixels

This is useful for tasks like image generation or upscaling, where you need to increase
the resolution of features that the network has processed.

## How It Works

An upsampling layer increases the spatial dimensions (height and width) of input tensors by repeating values from
the input to create a larger output. This implementation uses nearest-neighbor interpolation, which repeats each
value in the input tensor multiple times based on the scale factor to create the upsampled output.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UpsamplingLayer(Int32)` | Initializes a new instance of the `UpsamplingLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Int32)` | Calculates the output shape based on input shape and scale factor. |
| `Forward(Tensor<>)` | Performs the forward pass of the upsampling layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU tensors. |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves channel/spatial dims and computes output shape on first forward. |
| `ResetState` | Resets the internal state of the layer. |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuCachedInputShape` | Cached input shape from GPU forward pass for backward pass. |
| `_lastInput` | The input tensor from the last forward pass. |
| `_scaleFactor` | The factor by which to increase spatial dimensions. |

