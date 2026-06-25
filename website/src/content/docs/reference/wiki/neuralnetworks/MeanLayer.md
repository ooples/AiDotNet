---
title: "MeanLayer<T>"
description: "Represents a layer that computes the mean (average) of input values along a specified axis."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that computes the mean (average) of input values along a specified axis.

## For Beginners

This layer calculates the average of values in your data along one direction.

Think of it like calculating the average test score for each student across multiple subjects:

- Input: A table of scores where rows are students and columns are subjects
- MeanLayer with axis=1 (columns): Gives each student's average score across all subjects

Some practical examples:

- In image processing: Taking the average across color channels
- In text analysis: Taking the average of word embeddings to get a sentence representation
- In time series: Taking the average across time steps to get a summary

For instance, if you have data with shape [10, 5, 20] (e.g., 10 batches, 5 time steps, 20 features),
a MeanLayer with axis=1 would output shape [10, 20], giving you the average across all time steps.

## How It Works

The MeanLayer reduces the dimensionality of data by taking the average of values along a specified axis.
This operation is useful for aggregating feature information or reducing sequence data to a fixed-size
representation. The output shape has one fewer dimension than the input shape, with the specified axis
being removed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeanLayer(Int32)` | Initializes a new instance of the `MeanLayer` class with the specified input shape and axis. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Axis` | Gets the axis along which the mean is calculated. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Int32)` | Calculates the output shape of the mean layer based on the input shape and axis. |
| `Forward(Tensor<>)` | Performs the forward pass of the mean layer. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass for mean reduction. |
| `GetMetadata` | Gets all trainable parameters from the mean layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves shape on first forward by collapsing the axis dim from input.Shape. |
| `ResetState` | Resets the internal state of the mean layer. |
| `UpdateParameters()` | Updates the parameters of the mean layer using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuCachedInputShape` | Cached input shape for GPU backward pass. |
| `_lastInput` | The input tensor from the most recent forward pass. |
| `_lastOutput` | The output tensor from the most recent forward pass. |

