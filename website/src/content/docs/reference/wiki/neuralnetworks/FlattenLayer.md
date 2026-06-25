---
title: "FlattenLayer<T>"
description: "Represents a flatten layer that reshapes multi-dimensional input data into a 1D vector."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a flatten layer that reshapes multi-dimensional input data into a 1D vector.

## For Beginners

A flatten layer converts multi-dimensional data into a simple list of numbers.

Imagine you have a 2D grid of numbers (like a small image):
```
[
[1, 2, 3],
[4, 5, 6]
]
```

The flatten layer turns this into a single row:
```
[1, 2, 3, 4, 5, 6]
```

This transformation is needed because:

- Convolutional layers work with 2D or 3D data (like images)
- Fully connected layers expect a simple list of numbers
- Flatten layers bridge these two types of layers

Think of it like taking a book (a 3D object with pages) and reading all the text 
in order from beginning to end (a 1D sequence). All the information is preserved,
but it's rearranged into a different shape.

## How It Works

A flatten layer transforms multi-dimensional input data (such as images or feature maps) into a one-dimensional
vector. This is often necessary when transitioning from convolutional layers to fully connected layers
in a neural network. The flatten operation preserves all values and their order, just changing the way
they are arranged from a multi-dimensional tensor to a single vector.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU using a zero-copy view reshape. |
| `GetParameters` | Gets the trainable parameters of the layer. |
| `OnFirstForward(Tensor<>)` | Resolves shape on first forward by reading input.Shape and computing the flattened output size. |
| `ResetState` | Resets the internal state of the layer. |
| `UpdateParameters()` | Updates the parameters of the layer based on the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputShape` | The shape of the input tensor. |
| `_lastInput` | The input tensor from the last forward pass, saved for backpropagation. |
| `_outputSize` | The size of the output vector. |

