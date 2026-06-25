---
title: "BidirectionalLayer<T>"
description: "Represents a bidirectional layer that processes input sequences in both forward and backward directions."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a bidirectional layer that processes input sequences in both forward and backward directions.

## For Beginners

This layer looks at input data in two ways at the same time - both forward and backward.

Think of it like reading a sentence:

- Forward reading: "The cat sat on the mat" (left to right)
- Backward reading: "mat the on sat cat The" (right to left)

By processing data in both directions:

- The layer can understand context from both past and future elements
- It can discover patterns that might be missed if only looking in one direction
- It often improves performance on sequence tasks like text processing

For example, in the sentence "The bank is by the river", the meaning of "bank" 
depends on both previous words ("The") and future words ("by the river").
A bidirectional layer helps capture these relationships.

## How It Works

A bidirectional layer processes input sequences in two directions: forward (from first to last) and backward 
(from last to first). This approach allows the layer to capture patterns that depend on both past and future 
context. The outputs from both directions can either be merged (typically added together) or kept separate, 
depending on the configuration.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BidirectionalLayer(LayerBase<>,Boolean,IActivationFunction<>,IEngine)` | Initializes a new instance of the `BidirectionalLayer` class with the specified inner layer and a ReLU activation function. |
| `BidirectionalLayer(LayerBase<>,Boolean,IVectorActivationFunction<>,IEngine)` | Initializes a new instance of the `BidirectionalLayer` class with the specified inner layer and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | The computation engine (CPU or GPU) for vectorized operations. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU-accelerated forward pass. |
| `SupportsGpuTraining` | Gets a value indicating whether this layer supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Boolean)` | Calculates the output shape of the bidirectional layer based on the inner layer's output shape and merge mode. |
| `Forward(Tensor<>)` | Performs the forward pass of the bidirectional layer. |
| `ForwardGpu(Tensor<>[])` | Performs a GPU-resident forward pass of the bidirectional layer. |
| `GetParameters` | Gets all trainable parameters from both the forward and backward layers as a single vector. |
| `MergeOutputs(Tensor<>,Tensor<>)` | Merges the outputs from the forward and backward passes according to the configured merge mode. |
| `MergeOutputsGpu(IDirectGpuBackend,Tensor<>,Tensor<>,Int32,Int32)` | Merges forward and backward outputs on the GPU according to the merge mode. |
| `ResetState` | Resets the internal state of the bidirectional layer and its inner layers. |
| `ReverseSequence(Tensor<>)` | Reverses the sequence order along the time dimension (typically dimension 1). |
| `ReverseSequenceGpu(IDirectGpuBackend,Tensor<>,Int32,Int32,Int32)` | Reverses a sequence along the time dimension on the GPU. |
| `Serialize(BinaryWriter)` | Sets the trainable parameters for both the forward and backward layers. |
| `UpdateParameters()` | Updates the parameters of both forward and backward layers using the calculated gradients. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Performs the backward pass on GPU tensors. |

