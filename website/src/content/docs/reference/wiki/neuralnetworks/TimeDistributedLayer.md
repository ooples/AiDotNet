---
title: "TimeDistributedLayer<T>"
description: "Represents a wrapper layer that applies an inner layer to each time step of a sequence independently."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a wrapper layer that applies an inner layer to each time step of a sequence independently.

## For Beginners

This layer helps process sequences of data by applying the same operation to each step.

Think of it like an assembly line worker who performs the same task on each item that passes by:

- You have a sequence of items (like frames in a video or words in a sentence)
- You want to apply the same operation to each item independently
- This layer automates that process while preserving the original sequence order

For example, if you have a video with 30 frames per second, and you want to detect objects in each frame:

- A normal layer would need to process all frames together
- This time distributed layer would apply your object detection layer to each frame separately
- The result would be object detections for each frame, still organized as a sequence

This makes it much easier to work with sequential data like videos, sentences, or time series.

## How It Works

A time distributed layer applies the same inner layer (and its operations) to each time step of a sequence 
independently. This is particularly useful for processing sequential data where the same transformation needs 
to be applied to each element in the sequence. The layer maintains the temporal structure of the data while 
allowing each time step to be processed by the inner layer.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeDistributedLayer(LayerBase<>,IActivationFunction<>,Int32[])` | Initializes a new instance of the `TimeDistributedLayer` class with scalar activation function. |
| `TimeDistributedLayer(LayerBase<>,IVectorActivationFunction<>,Int32[])` | Initializes a new instance of the `TimeDistributedLayer` class with vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateInputShape(LayerBase<>,Int32[])` | Calculates the input shape of the time distributed layer. |
| `CalculateOutputShape(LayerBase<>,Int32[])` | Calculates the output shape of the time distributed layer. |
| `Forward(Tensor<>)` | Performs the forward pass of the time distributed layer. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetMetadata` | Persists the inner layer's type name + shape so DeserializationHelper can reconstruct the wrapped layer concretely. |
| `GetParameters` | Gets all trainable parameters of the inner layer. |
| `ResetState` | Resets the internal state of the layer and its inner layer. |
| `UpdateParameters()` | Updates the parameters of the inner layer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_accumulatedGradients` | Performs the backward pass using manual gradient computation. |
| `_innerLayer` | The inner layer that is applied to each time step. |
| `_lastInput` | The input tensor from the last forward pass. |
| `_lastOutput` | The output tensor from the last forward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

