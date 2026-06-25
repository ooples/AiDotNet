---
title: "AveragePoolingLayer<T>"
description: "Implements an average pooling layer for neural networks, which reduces the spatial dimensions of the input by taking the average value in each pooling window."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements an average pooling layer for neural networks, which reduces the spatial dimensions
of the input by taking the average value in each pooling window.

## How It Works

**For Beginners:** An average pooling layer helps reduce the size of data flowing through a neural network
while preserving overall characteristics. It works by dividing the input into small windows
(determined by the pool size) and computing the average of all values in each window.

Think of it like creating a lower-resolution summary: instead of keeping every detail,
you average all the values in each area to get a representative value.

This helps the network:

1. Preserve background information and overall context
2. Reduce computation needs
3. Smooth out noisy features

Average pooling is often used in the final layers of a network or when you want to
preserve more spatial information compared to max pooling.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AveragePoolingLayer(Int32,Int32)` | Creates a new average pooling layer with the specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PoolSize` | Gets the size of the pooling window. |
| `Strides` | Gets the step size when moving the pooling window across the input. |
| `SupportsGpuExecution` | Indicates that this layer supports GPU-accelerated execution. |
| `SupportsTraining` | Indicates whether this layer supports training operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Int32,Int32)` | Calculates the output shape based on the input shape and pooling parameters. |
| `Deserialize(BinaryReader)` | Loads the layer's configuration from a binary stream. |
| `Forward(Tensor<>)` | Performs the forward pass of the average pooling operation. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-resident forward pass of average pooling, keeping all data on GPU. |
| `GetActivationTypes` | Returns the activation functions used by this layer. |
| `GetParameters` | Gets all trainable parameters of the layer. |
| `GetPoolSize` | Gets the pool size as a 2D array (height, width). |
| `GetStride` | Gets the stride as a 2D array (height stride, width stride). |
| `OnFirstForward(Tensor<>)` | Resolves input shape on first forward (PyTorch AvgPool2d-style). |
| `ResetState` | Resets the internal state of the layer. |
| `Serialize(BinaryWriter)` | Saves the layer's configuration to a binary stream. |
| `UpdateParameters()` | Updates the layer's parameters during training. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_addedBatchDimension` | Tracks whether a batch dimension was added during the forward pass. |
| `_gpuInputShape` | Stores the input shape from GPU forward pass for backward pass. |
| `_lastInput` | Stores the last input tensor from the forward pass for use in autodiff backward pass. |
| `_lastOutputShape` | Stores the output shape for backward pass gradient distribution. |

