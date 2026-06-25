---
title: "InputLayer<T>"
description: "Represents an input layer that passes input data through unchanged to the next layer in the neural network."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents an input layer that passes input data through unchanged to the next layer in the neural network.

## For Beginners

This layer is like the doorway to your neural network.

Think of the InputLayer as:

- The entrance where your data first enters the neural network
- A way to tell the network what shape your data has
- A pass-through that doesn't change your data

For example, if you're processing images that are 28x28 pixels, you would use an InputLayer
with inputSize=784 (28×28) to tell the network about the size of each image.

Unlike other layers, the InputLayer doesn't learn or transform anything - it just
passes your data into the network.

## How It Works

The Input Layer serves as the entry point for data into a neural network. Unlike other layers, 
it doesn't transform the data or learn any parameters; it simply validates and passes the input
through to the next layer. This layer establishes the dimensionality of the input data for the
entire network.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InputLayer(Int32)` | Initializes a new instance of the `InputLayer` class with the specified input size. |
| `InputLayer(Int32[])` | Initializes a new instance of the `InputLayer` class with an explicit multi-dimensional input shape. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the input layer, simply returning the input unchanged. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetParameters` | Returns an empty vector since the input layer has no trainable parameters. |
| `ResetState` | Reset state is a no-op for the input layer since it maintains no state. |
| `UpdateParameters()` | Update parameters is a no-op for the input layer since it has no trainable parameters. |

