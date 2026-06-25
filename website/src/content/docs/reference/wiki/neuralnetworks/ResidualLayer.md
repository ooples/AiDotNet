---
title: "ResidualLayer<T>"
description: "Represents a residual layer that adds the identity mapping (input) to the output of an inner layer."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a residual layer that adds the identity mapping (input) to the output of an inner layer.

## For Beginners

This layer helps neural networks learn more effectively, especially when they're very deep.

Think of it as a "correction mechanism":

- The inner layer tries to learn how to improve or adjust the input
- The original input is preserved and added back in at the end
- This allows the network to focus on learning the changes needed, rather than recreating the entire signal

Benefits include:

- Solves the "vanishing gradient problem" that makes deep networks hard to train
- Enables training of much deeper networks (hundreds of layers instead of just dozens)
- Improves learning speed and accuracy

For example, in image recognition, a residual layer might learn to emphasize important features
while preserving the original image information through the skip connection.

## How It Works

A residual layer implements the core concept of residual networks (ResNets), where the layer learns
the residual (difference) between the identity mapping and the desired underlying mapping rather than
the complete transformation. This is achieved by adding a skip connection that passes the input directly
to the output, where it's added to the transformed output of an inner layer.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ResidualLayer(ILayer<>,IActivationFunction<>)` | Initializes a new instance of the `ResidualLayer` class with the specified input shape, inner layer, and scalar activation function. |
| `ResidualLayer(ILayer<>,IVectorActivationFunction<>)` | Initializes a new instance of the `ResidualLayer` class with the specified input shape, inner layer, and vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsShapeResolved` | A residual block is only fully shape-resolved once its WRAPPED inner layer is. |
| `ParameterCount` |  |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training through backpropagation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the residual layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass of the residual layer on the GPU. |
| `GetMetadata` | Returns metadata for serialization including inner layer dimensions. |
| `GetParameterGradients` |  |
| `OnFirstForward(Tensor<>)` | Resolves shape on first forward; output equals input shape (skip connection). |
| `ResetState` | Resets the internal state of the residual layer and its inner layer. |
| `SetInnerLayer(ILayer<>)` | Sets a new inner layer for the residual layer. |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` | Gets all trainable parameters from the inner layer as a single vector. |
| `UpdateParameters()` | Updates the parameters of the inner layer using the calculated gradients. |
| `ValidateInnerLayer` | Validates that the inner layer has the same input and output shape. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_innerLayer` | The inner layer that transforms the input before being added back to the original input. |
| `_lastInnerOutput` | Stores the output tensor from the inner layer during the forward pass. |
| `_lastInput` | Stores the input tensor from the most recent forward pass. |

