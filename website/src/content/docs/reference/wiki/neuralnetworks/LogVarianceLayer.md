---
title: "LogVarianceLayer<T>"
description: "Represents a layer that computes the logarithm of variance along a specified axis in the input tensor."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that computes the logarithm of variance along a specified axis in the input tensor.

## For Beginners

This layer measures how much the values in your data spread out from their average (variance),
and then takes the logarithm of that spread.

Think of it like measuring how consistent or varied your data is:

- Low values mean the data points are very similar to each other
- High values mean the data points vary widely

For example, if you have a set of images:

- Images that are very similar would produce low log-variance
- Images that are very different would produce high log-variance

This is often used in AI models that need to understand the variation in the data,
such as in models that generate new data similar to what they've been trained on.

## How It Works

The LogVarianceLayer calculates the statistical variance of values along a specified axis of the input tensor,
and then computes the natural logarithm of that variance. This is often used in neural networks for calculating
statistical measures, normalizing data, or as part of variational autoencoders (VAEs).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LogVarianceLayer(Int32)` | Initializes a new instance of the `LogVarianceLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Axis` | Gets the axis along which the variance is calculated. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training through backpropagation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Int32)` | Calculates the output shape of the log-variance layer based on the input shape and the axis along which variance is calculated. |
| `Forward(Tensor<>)` | Performs the forward pass of the log-variance layer. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass for log-variance reduction. |
| `GetMetadata` | Gets all trainable parameters of the layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves shape on first forward by collapsing the axis dim from input.Shape. |
| `ResetState` | Resets the internal state of the layer. |
| `UpdateParameters()` | Updates the parameters of the layer based on the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastInput` | The input tensor from the last forward pass. |
| `_lastOutput` | The output tensor from the last forward pass. |
| `_meanValues` | The mean values calculated during the last forward pass. |

