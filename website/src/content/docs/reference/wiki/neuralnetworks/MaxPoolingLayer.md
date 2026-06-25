---
title: "MaxPoolingLayer<T>"
description: "Implements a max pooling layer for neural networks, which reduces the spatial dimensions of the input by taking the maximum value in each pooling window."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a max pooling layer for neural networks, which reduces the spatial dimensions
of the input by taking the maximum value in each pooling window.

## How It Works

**For Beginners:** A max pooling layer helps reduce the size of data flowing through a neural network
while keeping the most important information. It works by dividing the input into small windows
(determined by the pool size) and keeping only the largest value from each window.

Think of it like summarizing a detailed picture: instead of describing every pixel,
you just point out the most noticeable feature in each area of the image.

This helps the network:

1. Focus on the most important features
2. Reduce computation needs
3. Make the model more robust to small changes in input position

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaxPoolingLayer(Int32,Int32)` | Creates a new max pooling layer with the specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PoolSize` | Gets the size of the pooling window. |
| `Stride` | Gets the step size when moving the pooling window across the input. |
| `SupportsGpuExecution` | Indicates that this layer supports GPU-accelerated execution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Int32,Int32)` | Calculates the output shape based on the input shape and pooling parameters. |
| `Deserialize(BinaryReader)` | Loads the layer's configuration from a binary stream. |
| `Forward(Tensor<>)` | Performs the forward pass of the max pooling operation. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-resident forward pass of max pooling, keeping all data on GPU. |
| `GetActivationTypes` | Returns the activation functions used by this layer. |
| `GetParameters` | Gets all trainable parameters of the layer. |
| `GetPoolSize` | Indicates whether this layer supports training operations. |
| `GetStride` | Gets the stride for the pooling operation. |
| `OnFirstForward(Tensor<>)` | Resolves input shape on first forward (PyTorch MaxPool2d-style). |
| `ResetState` | Resets the internal state of the layer. |
| `Serialize(BinaryWriter)` | Saves the layer's configuration to a binary stream. |
| `UpdateParameters()` | Updates the layer's parameters during training. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_addedBatchDimension` | Tracks whether a batch dimension was added during the forward pass. |
| `_gpuIndices` | Stores GPU-resident pooling indices for backward pass. |
| `_gpuInputShape` | Stores the input shape from the GPU forward pass for backward pass. |
| `_lastInput` | Stores the last input tensor from the forward pass for use in autodiff backward pass. |
| `_lastOutputShape` | Stores the actual output shape for 3D inputs (may differ from pre-computed OutputShape). |
| `_maxIndices` | Stores the indices of the maximum values found during the forward pass. |

