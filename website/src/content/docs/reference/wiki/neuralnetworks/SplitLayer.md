---
title: "SplitLayer<T>"
description: "Represents a layer that splits the input tensor along a specific dimension into multiple equal parts."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that splits the input tensor along a specific dimension into multiple equal parts.

## For Beginners

This layer breaks up your input data into smaller, equal-sized chunks.

Think of it like cutting a pizza into equal slices:

- Your input data is the whole pizza
- The number of splits determines how many slices you want
- Each slice has the same size and shape

Benefits include:

- Processing different parts of the input in parallel
- Allowing different operations on different parts of the input
- Creating multi-stream architectures where each stream handles a portion of the data

For example, in natural language processing, you might split word embeddings to create
multiple "attention heads" that each focus on different aspects of the text.

## How It Works

A split layer divides the input tensor into multiple equal parts along a specified dimension. This is useful
for parallel processing of data or for implementing multi-headed attention mechanisms. The layer ensures that 
the input size is divisible by the number of splits to maintain consistency.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SplitLayer(Int32)` | Initializes a new instance of the `SplitLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training through backpropagation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Int32)` | Calculates the output shape of the split layer based on input shape and number of splits. |
| `Forward(Tensor<>)` | Performs the forward pass of the split layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors. |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves shape on first forward; output adds a leading dim of size numSplits and divides last dim. |
| `ResetState` | Resets the internal state of the split layer. |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastInput` | Stores the input tensor from the most recent forward pass. |
| `_numSplits` | The number of parts to split the input tensor into. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

