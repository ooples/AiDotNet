---
title: "MessagePassingLayer<T>"
description: "Implements a general Message Passing Neural Network (MPNN) layer."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a general Message Passing Neural Network (MPNN) layer.

## For Beginners

Think of message passing like spreading information through a network.

Imagine a social network where:

1. **Message**: Each friend sends you a message (combining their info with yours)
2. **Aggregate**: You collect and summarize all messages from friends
3. **Update**: You update your own status based on the summary

This happens for all people simultaneously, allowing information to flow through the network.

Use cases:

- Molecule analysis: Atoms sharing information about chemical bonds
- Social networks: Users influenced by their connections
- Citation networks: Papers learning from papers they cite
- Recommendation systems: Items learning from similar items

## How It Works

Message Passing Neural Networks provide a general framework for graph neural networks.
The framework consists of three key functions:

1. Message: Computes messages from neighbors
2. Aggregate: Combines messages from all neighbors
3. Update: Updates node representations using aggregated messages

The layer performs the following computation for each node v:

- m_v = AGGREGATE({MESSAGE(h_u, h_v, e_uv) : u ∈ N(v)})
- h_v' = UPDATE(h_v, m_v)

where h_v are node features, e_uv are edge features, and N(v) is the neighborhood of v.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MessagePassingLayer(Int32,Int32,Int32,Boolean,Int32,IActivationFunction<>)` | Initializes a new instance of the `MessagePassingLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputFeatures` |  |
| `OutputFeatures` |  |
| `ParameterCount` |  |
| `SupportsGpuExecution` | Gets whether this layer supports GPU execution. |
| `UsesSparseAggregation` | Gets whether sparse (edge-based) aggregation is currently enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearEdges` | Clears the edge list and switches back to dense adjacency matrix aggregation. |
| `ClearGpuCache` | Clears GPU cache tensors and gradients. |
| `ClearGradients` |  |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass for Message Passing Neural Network. |
| `GetAdjacency(Int32,Int32,Int32)` | Helper to get adjacency value - supports both 2D [nodes, nodes] and 3D [batch, nodes, nodes]. |
| `GetAdjacencyMatrix` |  |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameterTensors` | Gets all trainable parameters as a list of tensors. |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes layer parameters using Xavier initialization. |
| `InitializeTensor(Tensor<>,)` | Initializes a tensor with scaled random values. |
| `PadOrSliceLastAxis(Tensor<>,Int32,Int32)` | Resizes the last axis of a 2D `[rows, fromWidth]` tensor to `toWidth`: slices when toWidth < fromWidth, zero-pads when toWidth > fromWidth, returns the input unchanged when equal. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetAdjacencyMatrix(Tensor<>)` |  |
| `SetEdgeFeatures(Tensor<>)` | Sets the edge features tensor. |
| `SetEdges(Tensor<Int32>,Tensor<Int32>)` | Sets the edge list representation of the graph structure for sparse aggregation. |
| `SetParameterTensors(List<Tensor<>>)` | Sets all trainable parameters from a list of tensors. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adjacencyMatrix` | The adjacency matrix defining graph structure. |
| `_edgeFeatures` | Edge features tensor (optional). |
| `_edgeSourceIndices` | Edge source node indices for sparse graph representation. |
| `_edgeTargetIndices` | Edge target node indices for sparse graph representation. |
| `_edgeWeights` | Edge feature transformation weights (optional). |
| `_lastAggregated` | Cached aggregated messages. |
| `_lastInput` | Cached input from forward pass. |
| `_lastMessageHidden` | Cached hidden activations from message MLP layer 1. |
| `_lastMessages` | Cached messages for backward pass. |
| `_lastOutput` | Cached output from forward pass. |
| `_lastResetGate` | Cached reset gates. |
| `_lastUpdateGate` | Cached update gates. |
| `_messageWeights1` | Message computation network (MLP). |
| `_messageWeights1Gradient` | Gradients for parameters. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_resetWeights` | Reset gate weights (GRU-style). |
| `_updateWeights` | Update computation network (GRU-style update). |
| `_useSparseAggregation` | Indicates whether to use sparse (edge-based) or dense (adjacency matrix) aggregation. |

