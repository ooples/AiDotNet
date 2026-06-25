---
title: "ConcatenateLayer<T>"
description: "Represents a neural network layer that concatenates multiple inputs along a specified axis."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a neural network layer that concatenates multiple inputs along a specified axis.

## For Beginners

A concatenate layer joins multiple inputs together to make one bigger output.

Think of it like joining arrays or lists:

- If you have two lists [1, 2, 3] and [4, 5], concatenating them gives [1, 2, 3, 4, 5]

In neural networks, we often work with multi-dimensional data, so we need to specify which
dimension (axis) to join along:

- Axis 0 would join along the first dimension (like stacking sheets of paper)
- Axis 1 would join along the second dimension (like extending rows sideways)
- Axis 2 would join along the third dimension (like extending columns downward)

For example, if you have:

- One tensor representing features from an image: [batch_size, 100]
- Another tensor representing features from text: [batch_size, 50]

You could use a concatenate layer with axis=1 to create a combined feature tensor of shape [batch_size, 150]
that contains both sets of features side by side.

## How It Works

A concatenate layer combines multiple input tensors into a single output tensor by joining them along
a specified axis. For example, if you have two tensors of shape [batch_size, 10] and [batch_size, 15],
concatenating them along axis 1 would produce a tensor of shape [batch_size, 25]. This layer doesn't
have any trainable parameters and simply passes the gradients back to the appropriate input tensors
during backpropagation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConcatenateLayer(Int32[][],Int32,IActivationFunction<>)` | Initializes a new instance of the `ConcatenateLayer` class with a scalar activation function. |
| `ConcatenateLayer(Int32[][],Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `ConcatenateLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` |  |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[][],Int32)` | Calculates the output shape of the concatenated tensor. |
| `ComputeActivationBackwardGpu(DirectGpuTensorEngine,Tensor<>,Tensor<>,FusedActivationType)` | Computes the activation backward gradient on GPU. |
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>)` | This method is not supported by ConcatenateLayer and will throw an exception. |
| `Forward(Tensor<>[])` | Performs the forward pass of the concatenate layer with multiple inputs. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetParameters` | Gets all trainable parameters from the layer as a single vector. |
| `ResetState` | Resets the internal state of the concatenate layer. |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |
| `ValidateInputShapes(Int32[][])` | Validates that the input shapes are compatible for concatenation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputPortsCache` | Declares named input ports for this multi-input layer. |

