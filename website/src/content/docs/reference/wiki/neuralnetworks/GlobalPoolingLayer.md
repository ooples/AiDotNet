---
title: "GlobalPoolingLayer<T>"
description: "Represents a global pooling layer that reduces spatial dimensions to a single value per channel."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a global pooling layer that reduces spatial dimensions to a single value per channel.

## For Beginners

A global pooling layer summarizes each feature map into a single value.

Imagine you have a set of 2D feature maps (like heat maps showing where different features appear):

- Global pooling looks at each entire feature map
- It creates a single number that represents that entire feature map
- This dramatically reduces the amount of data while preserving the most important information

For example, with 64 feature maps of size 7×7:

- Input: 7×7—64 (3,136 values)
- Output: 1×1—64 (64 values, one per feature map)

There are two main types of global pooling:

- Global Max Pooling: Takes the maximum value from each feature map

(useful for detecting if a feature appears anywhere in the input)

- Global Average Pooling: Takes the average of all values in each feature map

(useful for determining the overall presence of a feature)

Global pooling is often used as the final layer before classification,
replacing large fully connected layers and reducing overfitting.

## How It Works

A global pooling layer reduces the spatial dimensions (height and width) of the input feature maps
to a single value per channel. This is achieved by applying a pooling operation (such as max or average)
across the entire spatial extent of each channel. Global pooling is often used at the end of convolutional
neural networks to reduce the spatial dimensions before connecting to fully connected layers, providing
some translation invariance and reducing the number of parameters.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GlobalPoolingLayer(PoolingType,IVectorActivationFunction<>)` | Initializes a new instance of the `GlobalPoolingLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyScalarActivationAutodiff(ComputationNode<>)` | Applies scalar activation function with autodiff support. |
| `CalculateOutputShape(Int32[])` | Calculates the output shape of the global pooling layer based on the input shape. |
| `ComputeActivationBackwardGpu(DirectGpuTensorEngine,Tensor<>,Tensor<>,FusedActivationType)` | Computes the activation backward gradient on GPU. |
| `CreateHighDimensionalOutputShape(Int32[])` | Creates output shape for tensors with more than 4 dimensions. |
| `Forward(Tensor<>)` | Performs the forward pass of the global pooling layer. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass using GPU-resident tensors. |
| `GetParameters` | Gets the trainable parameters of the layer. |
| `GetReductionAxes(Int32)` | Gets the axes to reduce over based on input tensor rank. |
| `OnFirstForward(Tensor<>)` | Resolves input shape on first forward (PyTorch-style). |
| `ResetState` | Resets the internal state of the layer. |
| `UpdateParameters()` | Updates the parameters of the layer based on the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastInput` | The input tensor from the last forward pass, saved for backpropagation. |
| `_lastOutput` | The output tensor from the last forward pass, saved for backpropagation. |
| `_maxIndices` | Stores the indices of the maximum values found during global max pooling. |
| `_poolingType` | The type of pooling operation to apply globally (Max or Average). |

