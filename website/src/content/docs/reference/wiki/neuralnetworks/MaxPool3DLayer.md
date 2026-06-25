---
title: "MaxPool3DLayer<T>"
description: "Represents a 3D max pooling layer for downsampling volumetric data."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a 3D max pooling layer for downsampling volumetric data.

## For Beginners

Max pooling works like summarizing a 3D region by keeping only
the largest value.

Think of it like this:

- You have a 3D grid of numbers
- You divide it into small cubes (e.g., 2x2x2)
- For each cube, you keep only the largest number
- This makes your data smaller while keeping the important features

This is useful because:

- It reduces the amount of computation needed
- It helps the network focus on the most important features
- It makes the network more robust to small position changes

## How It Works

A 3D max pooling layer reduces the spatial dimensions (depth, height, width) of volumetric
data while preserving the most prominent features. This helps reduce computational cost
and provides translation invariance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaxPool3DLayer(Int32,Int32)` | Initializes a new instance of the `MaxPool3DLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PoolSize` | Gets the size of the pooling window (same for depth, height, width). |
| `Stride` | Gets the stride (step size) for moving the pooling window. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training (backpropagation). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(BinaryReader)` | Deserializes the layer from a binary stream. |
| `Forward(Tensor<>)` | Performs the forward pass of 3D max pooling. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-resident forward pass of 3D max pooling, keeping all data on GPU. |
| `GetActivationTypes` | Gets the activation function types used by this layer. |
| `OnFirstForward(Tensor<>)` | Resolves channel/spatial dims and registers the resolved output shape on first forward. |
| `ResetState` | Resets the cached state from forward/backward passes. |
| `Serialize(BinaryWriter)` | Serializes the layer to a binary stream. |
| `UpdateParameters()` | Updates parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_addedBatchDimension` | Whether batch dimension was added in ForwardGpu. |
| `_gpuIndicesBuffer` | GPU buffer containing pooling indices for backward pass. |
| `_gpuInputShape` | Cached GPU input shape for backward pass. |
| `_lastInput` | Cached input from the last forward pass. |
| `_maxIndices` | Stores the indices of maximum values from the forward pass for gradient routing. |

