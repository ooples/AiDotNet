---
title: "ParallelStreamsLayer<T>"
description: "Splits input along the feature axis into two equal halves, processes each half through its own independent sub-network (stream), and concatenates the two stream outputs."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Splits input along the feature axis into two equal halves, processes each half through its own
independent sub-network (stream), and concatenates the two stream outputs.

## For Beginners

Many real-world neural networks need to process two different types of
information at the same time. For example:

- An audio-visual model processes audio features and video features separately
- A multi-modal model processes text embeddings and image embeddings in parallel
- A siamese network processes two inputs through shared or separate encoders

This layer solves the problem of expressing parallel processing within a sequential layer stack.
Instead of needing complex custom forward logic, you can simply place this layer in your network
and it will:

1. **Split** the input features into two equal halves (e.g., [256 audio | 256 visual])
2. **Process** each half through its own set of layers (Stream A and Stream B)
3. **Concatenate** the two outputs back together

All operations are tracked on the gradient tape so backpropagation works correctly through
both streams simultaneously.

**Example:** If you have 512 input features representing concatenated audio+visual features:

## How It Works

**How it works internally:**

- The input tensor is sliced along the last axis into two equal halves using

`Engine.TensorSlice` (tape-tracked)

- Each half is passed through its respective stream's layers sequentially
- The two stream outputs are concatenated along the last axis using

`Engine.TensorConcatenate` (tape-tracked)

**Gradient flow:** Because the split and concatenation use Engine operations that record on
the autodiff tape, gradients flow correctly through both streams during backpropagation.
Each stream receives only the gradients relevant to its portion of the input, ensuring proper
learning for both sub-networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ParallelStreamsLayer(Int32,Int32,Int32,IEnumerable<ILayer<>>,IEnumerable<ILayer<>>)` | Creates a parallel streams layer that splits input features and processes each half independently. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters across both streams. |
| `SupportsTraining` | Gets whether this layer supports training through backpropagation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass: splits input, runs both streams, and concatenates outputs. |
| `GetParameterGradients` | Collects parameter gradients from all layers in both streams. |
| `GetParameters` | Collects all trainable parameters from both streams into a single vector. |
| `ResetState` | Resets the internal state of all layers in both streams. |
| `RunStream(List<ILayer<>>,Tensor<>)` | Runs input through a sequence of layers, returning the final output. |
| `SetParameters(Vector<>)` | Sets all trainable parameters for both streams from a single parameter vector. |
| `SetTrainingMode(Boolean)` | Sets training or inference mode for all layers in both streams. |
| `UpdateParameters()` | Updates parameters in both streams using the given learning rate. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_splitSize` | Half the input feature dimension — the number of features each stream receives. |
| `_streamA` | The layers that process the first half of the input features. |
| `_streamB` | The layers that process the second half of the input features. |

