---
title: "CroppingLayer<T>"
description: "Represents a cropping layer that removes portions of input tensors from the edges."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a cropping layer that removes portions of input tensors from the edges.

## For Beginners

A cropping layer cuts off the edges of your data.

Think of it like cropping a photo:

- You can trim different amounts from the top, bottom, left, and right
- The middle portion (the important part) is kept
- The trimmed edges are discarded

For example, in image processing:

- You might crop off padding added by previous layers
- You might focus on the central region where the important features are
- You might adjust the size to match what the next layer expects

Cropping layers are simple but useful for controlling exactly what part of the data
flows through your neural network.

## How It Works

A cropping layer removes specified portions from the edges of an input tensor. This is useful for
removing border artifacts, adjusting dimensions between layers, or focusing on specific regions
of input data. The cropping can be applied differently to each dimension of the input.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CroppingLayer(Int32[],Int32[],Int32[],Int32[],IActivationFunction<>,IEngine)` | Initializes a new instance of the `CroppingLayer` class with the specified cropping parameters and a scalar activation function. |
| `CroppingLayer(Int32[],Int32[],Int32[],Int32[],IVectorActivationFunction<>,IEngine)` | Initializes a new instance of the `CroppingLayer` class with the specified cropping parameters and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` |  |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training through backpropagation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Int32[],Int32[],Int32[],Int32[])` | Calculates the output shape after applying the cropping operations. |
| `Forward(Tensor<>)` | Processes the input data through the cropping layer. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves the input/output shape on first forward by subtracting crops from input.Shape. |
| `ResetState` | Resets the internal state of the layer. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer from a single vector. |
| `UpdateParameters()` | Updates the layer's parameters using the specified learning rate. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cropBottom` | The amount to crop from the bottom of each dimension. |
| `_cropLeft` | The amount to crop from the left of each dimension. |
| `_cropRight` | The amount to crop from the right of each dimension. |
| `_cropTop` | The amount to crop from the top of each dimension. |
| `_gpuCachedInputShape` | Cached input shape for GPU backward pass. |
| `_lastInput` | Stores the last input for use in autodiff backward pass. |
| `_originalInputShape` | Original input shape for restoring higher-rank tensors after processing. |

