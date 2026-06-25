---
title: "PixelShuffleLayer<T>"
description: "Pixel shuffle (sub-pixel convolution) layer for efficient spatial upsampling."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Pixel shuffle (sub-pixel convolution) layer for efficient spatial upsampling.

## For Beginners

Imagine you have a small image and want to make it bigger.

Pixel shuffle works by:

1. Starting with extra channel information (4x more channels for 2x upscaling)
2. Rearranging those channel values into spatial positions
3. Creating a larger image with the same amount of total information

For example, with 2x upscaling:

- Input: 64 channels × 32×32 pixels
- Output: 16 channels × 64×64 pixels (same total data, different arrangement)

This is commonly used in super-resolution models like Real-ESRGAN and ESPCN.

## How It Works

Pixel shuffle rearranges elements from the channel dimension into spatial dimensions,
effectively upscaling the spatial resolution. This is more efficient than transposed
convolution (deconvolution) for upsampling operations.

For a 2x upscaling, the layer takes 4 channel values and arranges them as a 2x2 spatial block.
The operation follows the formula: [batch, channels * r^2, height, width] -> [batch, channels, height * r, width * r]

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PixelShuffleLayer(Int32)` | Initializes a new instance of the `PixelShuffleLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Indicates whether this layer supports GPU execution. |
| `SupportsTraining` |  |
| `UpscaleFactor` | Gets the upscaling factor used by this layer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateHighDimensionalOutputShape(Int32[],Int32)` | Calculates output shape for tensors with more than 5 dimensions. |
| `CalculateOutputShape(Int32[],Int32)` | Calculates the output shape based on input shape and upscale factor. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors. |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` | Resolves channel/spatial dims and registers the resolved output shape on first forward. |
| `ResetState` |  |
| `UpdateParameters()` |  |
| `ValidateInputShape(Int32[],Int32)` | Validates that the input shape is compatible with the upscale factor. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuAdded3DBatch` | Whether a batch dimension was added for 3D input in GPU forward. |
| `_gpuCachedInputShape` | Cached GPU input shape for backward pass. |
| `_lastInput` | Cached input from the last forward pass for backpropagation. |
| `_originalInputShape` | Cached original input shape for backward pass with higher-rank tensors. |
| `_upscaleFactor` | The upscaling factor for spatial dimensions. |

