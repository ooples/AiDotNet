---
title: "PoolingLayer<T>"
description: "Represents a layer that performs pooling operations on input tensors."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that performs pooling operations on input tensors.

## For Beginners

This layer helps reduce the size of your data while keeping the important information.

Think of it like creating a thumbnail of an image:

- The pooling layer divides your input into small regions (e.g., 2×2 squares)
- For each region, it either:
- Takes the maximum value (max pooling): good for detecting features like edges
- Takes the average value (average pooling): good for preserving background information
- This creates a smaller output with fewer pixels but retains the important features

For example, using 2×2 max pooling on a 4×4 image would give you a 2×2 output,
where each value is the maximum from its corresponding 2×2 region in the input.

Pooling helps make your neural network:

- More efficient (by reducing the amount of data)
- More robust (by being less sensitive to exact positions of features)
- Less prone to overfitting (by reducing the number of parameters)

## How It Works

The PoolingLayer reduces the spatial dimensions (height and width) of input tensors by applying
either max pooling or average pooling within local regions. This operation is commonly used in 
convolutional neural networks to reduce the spatial dimensions of feature maps, which helps to
reduce computation, provide translation invariance, and control overfitting.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PoolingLayer(Int32,Int32,PoolingType)` | Initializes a new instance of the `PoolingLayer` class with the specified dimensions and pooling parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PoolSize` | Gets the size of the pooling window. |
| `Stride` | Gets the stride of the pooling operation. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer has a GPU implementation. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |
| `Type` | Gets the type of pooling operation to perform. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputDimension(Int32,Int32,Int32)` | Calculates the output dimension size based on the input dimension, pool size, and stride. |
| `Forward(Tensor<>)` | Performs the forward pass of the pooling layer. |
| `GetMetadata` | Returns layer-specific metadata for serialization purposes. |
| `GetParameters` | Gets all trainable parameters from the pooling layer as a single vector. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the pooling layer. |
| `UpdateParameters()` | Updates the parameters of the pooling layer using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_addedBatchDimension` | Tracks whether a batch dimension was added during forward pass. |
| `_lastInput` | The input tensor from the most recent forward pass. |
| `_maxIndices` | The execution engine for GPU-accelerated pooling operations. |

