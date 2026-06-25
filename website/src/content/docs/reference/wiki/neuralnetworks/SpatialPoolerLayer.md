---
title: "SpatialPoolerLayer<T>"
description: "Represents a spatial pooler layer inspired by hierarchical temporal memory (HTM) principles."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a spatial pooler layer inspired by hierarchical temporal memory (HTM) principles.

## For Beginners

This layer helps convert input data into a format that's easier for neural networks to learn from.

Think of it like a translator that:

- Takes dense input information (where many values can be active)
- Converts it to a sparse representation (where only a few values are active)
- Preserves the important patterns and relationships in the data

Benefits include:

- Better handling of noisy or incomplete data
- More efficient representation of information
- Improved ability to recognize patterns

For example, when processing images, a spatial pooler might identify the most important features
while ignoring background noise or variations that don't matter for classification.

## How It Works

A spatial pooler is a key component in HTM systems that converts input patterns into sparse distributed
representations (SDRs). It maps input space to a new representation that captures the spatial structure
of the input while maintaining semantic similarity. The spatial pooler creates these representations
by selecting a small subset of active columns based on their connection strengths to the input.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpatialPoolerLayer(Int32,Double)` | Initializes a new instance of the `SpatialPoolerLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training through backpropagation. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the spatial pooler layer. |
| `ForwardGpu(Tensor<>[])` | Performs the GPU-accelerated forward pass for the spatial pooler layer. |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `InitializeConnections` | Initializes the connection strengths between input elements and columns with random values. |
| `Learn(Vector<>)` | Performs a learning step on the spatial pooler using the provided input. |
| `NormalizeConnections` | Normalizes the connection strengths to ensure all columns have balanced total connection strength. |
| `OnFirstForward(Tensor<>)` | Resolves input size from input.Shape (last axis or product of trailing axes) on first forward and allocates the connection matrix [inputSize, columnCount]. |
| `ResetState` | Resets the internal state of the spatial pooler layer. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer from a single vector. |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `BoostFactor` | The factor that controls boosting of inactive columns. |
| `ColumnCount` | The number of columns in the spatial pooler. |
| `Connections` | The connection strengths between input elements and columns. |
| `InputSize` | The size of the input vector. |
| `LastInput` | Stores the input tensor from the most recent forward pass or learning step. |
| `LastOutput` | Stores the output tensor from the most recent forward pass or learning step. |
| `LearningRate` | The learning rate used during the learning process. |
| `SparsityThreshold` | The threshold that determines the sparsity of the output. |
| `_connectionsGradient` | Gradient of the connections computed during backpropagation. |

