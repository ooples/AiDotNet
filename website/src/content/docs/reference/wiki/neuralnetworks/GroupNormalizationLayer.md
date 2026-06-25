---
title: "GroupNormalizationLayer<T>"
description: "Represents a Group Normalization layer that normalizes inputs across groups of channels."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Group Normalization layer that normalizes inputs across groups of channels.

## For Beginners

This layer helps stabilize training for convolutional networks.

Think of Group Normalization like organizing students into study groups:

- Each group (of channels) studies together and normalizes their behavior
- It works the same regardless of class size (batch size)
- This is especially useful for generative models like VAEs where batch sizes may be small

Key advantages:

- Works well with small batch sizes (even batch size of 1)
- More stable than Batch Normalization for generative models
- Used extensively in modern architectures like Stable Diffusion VAE

Typical usage:

- numGroups=32 for 256+ channels
- numGroups=16 for 128 channels
- numGroups=8 for 64 channels

## How It Works

Group Normalization divides channels into groups and normalizes the features within each group.
This makes it invariant to batch size, making it suitable for small batch sizes or applications
where batch statistics are not reliable (like VAEs and generative models).

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsGpuTraining` | Gets a value indicating whether this layer supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildGroupNormOptimizerState(String)` | Builds optimizer state for a specific parameter tensor. |
| `EnsureGroupNormOptimizerState(IGpuOptimizerConfig,IDirectGpuBackend)` | Ensures optimizer state buffers are allocated for the optimizer type. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | GPU-resident parameter update using the provided optimizer configuration. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_addedBatchDimension` | Tracks whether we added a batch dimension to a 3D input. |
| `_originalInputShape` | Original input shape for restoring higher-rank tensors after processing. |

