---
title: "ReshapeLayer<T>"
description: "Represents a reshape layer that transforms the dimensions of input data without changing its content."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a reshape layer that transforms the dimensions of input data without changing its content.

## For Beginners

This layer changes how your data is organized without changing the data itself.

Think of the ReshapeLayer like reorganizing a deck of playing cards:

- If you have cards arranged in 4 rows of 13 cards (representing the 4 suits)
- You could reorganize them into 13 rows of 4 cards (representing the 13 ranks)
- The cards themselves haven't changed, just how they're arranged

For example, in image processing:

- You might have an image of shape [height, width, channels]
- But a particular layer might need the data as a flat vector
- A reshape layer can convert between these formats without losing information

Common use cases include:

- Flattening data (e.g., converting a 2D image to a 1D vector for a dense layer)
- Reshaping for convolutional operations (e.g., turning a vector into a 3D tensor)
- Batch dimension manipulation (e.g., splitting or combining batch items)

The key requirement is that the total number of elements stays the same - you're just
reorganizing them into a different dimensional structure.

## How It Works

The ReshapeLayer rearranges the elements of the input tensor into a new shape without changing the data itself.
This operation is useful for connecting layers with different shape requirements or for preparing data for
specific layer types. The total number of elements must remain the same between the input and output shapes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReshapeLayer(Int32[])` | Initializes a new instance of the `ReshapeLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the reshape layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU using a zero-copy view reshape. |
| `GetParameters` | Gets all trainable parameters of the reshape layer as a single vector. |
| `GetTargetShape` | Gets the target shape for the reshape operation. |
| `OnFirstForward(Tensor<>)` | Resolves input shape on first forward; validates element-count compatibility with target output. |
| `ResetState` | Resets the internal state of the reshape layer. |
| `UpdateParameters()` | Updates the parameters of the reshape layer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuCachedInputShape` | Cached GPU input shape for backward pass. |
| `_inputShape` | The shape of the input tensor, excluding the batch dimension. |
| `_lastInput` | Stores the input tensor from the most recent forward pass for use in backpropagation. |
| `_outputShape` | The shape of the output tensor, excluding the batch dimension. |

