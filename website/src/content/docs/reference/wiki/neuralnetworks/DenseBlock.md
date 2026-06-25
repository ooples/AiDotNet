---
title: "DenseBlock<T>"
description: "Implements a Dense Block from the DenseNet architecture."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a Dense Block from the DenseNet architecture.

## For Beginners

Dense connectivity means each layer can directly access
features from all previous layers, promoting feature reuse and reducing
the need for redundant feature learning.

Key benefits:

- Strong gradient flow (helps with training very deep networks)
- Feature reuse (each layer can use features from all previous layers)
- Fewer parameters (layers can be narrow since they share features)

## How It Works

A Dense Block is the core building block of DenseNet. It contains multiple layers where
each layer receives feature maps from ALL preceding layers (dense connectivity).
This creates strong gradient flow and feature reuse throughout the network.

Architecture of a Dense Block with n layers:

Where k is the growth rate (number of channels added per layer).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DenseBlock(Int32,Int32,Double)` | Initializes a new instance of the `DenseBlock` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GrowthRate` | Gets the growth rate (channels added per layer). |
| `NumLayers` | Gets the number of layers in this dense block. |
| `OutputChannels` | Gets the number of output channels (inputChannels + numLayers × growthRate). |
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer has a GPU implementation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddGradients(Tensor<>,Tensor<>)` | Adds two gradient tensors of the same shape element-wise. |
| `ConcatenateChannels(Tensor<>,Tensor<>)` | Concatenates two tensors along the channel dimension (dim=0 for CHW, dim=1 for NCHW). |
| `Forward(Tensor<>)` | Performs the forward pass of the Dense Block. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU, keeping data GPU-resident. |
| `GetParameters` | Gets all trainable parameters from the block. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the block. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from the given parameter vector. |
| `SplitGradient(Tensor<>,Int32,Int32)` | Splits a gradient tensor along the channel dimension (dim=0 for CHW, dim=1 for NCHW). |
| `UpdateParameters()` | Updates the parameters of all sub-layers. |

