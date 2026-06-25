---
title: "GraphAttentionLayer<T>"
description: "Implements Graph Attention Network (GAT) layer for processing graph-structured data with attention mechanisms."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements Graph Attention Network (GAT) layer for processing graph-structured data with attention mechanisms.

## How It Works

Graph Attention Networks (GAT) introduced by Veličković et al. use attention mechanisms to learn
the relative importance of neighboring nodes. Unlike standard GCN which treats all neighbors equally,
GAT can assign different weights to different neighbors, allowing the model to focus on the most
relevant connections. The layer uses multi-head attention for robustness and expressiveness.

The attention mechanism computes: α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
where α_ij is the attention coefficient from node j to node i, W is a weight matrix,
h_i and h_j are node features, a is the attention vector, and || denotes concatenation.

**Production-Ready Features:**

- Fully vectorized operations using IEngine for GPU acceleration
- Tensor-based weights for all parameters
- Dual backward pass: BackwardManual() for efficiency, BackwardViaAutodiff() for accuracy
- Full gradient computation through attention mechanism
- Complete GetParameters()/SetParameters() for model persistence

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphAttentionLayer(Int32,Int32,Int32,Double,Double,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `GraphAttentionLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the LeakyReLU negative-slope used inside the attention score computation. |
| `DropoutRate` | Gets the dropout rate applied to attention coefficients during training. |
| `InputFeatures` |  |
| `NumHeads` | Gets the number of attention heads used in multi-head attention. |
| `OutputFeatures` |  |
| `ParameterCount` |  |
| `SupportsGpuExecution` | Gets whether this layer supports GPU execution. |
| `SupportsTraining` |  |
| `UsesSparseAggregation` | Gets whether sparse (edge-based) aggregation is currently enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add2DSliceTo3D(Tensor<>,Int32,Tensor<>)` | Adds a 2D tensor to a 3D tensor at position [batchIdx, :, :] in-place. |
| `Add2DSliceTo3DHead(Tensor<>,Int32,Tensor<>)` | Adds a 2D tensor to a 3D tensor slice [headIdx, :, :] in-place. |
| `BatchedMatMul3Dx2D(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32)` | Performs batched matrix multiplication for 3D × 2D tensors. |
| `ClearEdges` | Clears the edge list and switches back to dense adjacency matrix aggregation. |
| `ClearGpuCache` | Clears GPU cache tensors and gradients. |
| `ComputeLeakyReluGradientMatrix(Tensor<>,Tensor<>)` | Computes the LeakyReLU gradient matrix: 1 if preSoftmax > 0, else alpha. |
| `ComputeWeightGradientsViaEngine(Tensor<>,Int32,Int32,)` | Computes weight gradients using vectorized Engine operations as a fallback. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass for Graph Attention Networks. |
| `Get2DSliceFrom4D(Tensor<>,Int32,Int32)` | Extracts a 2D slice from a 4D tensor at position [batchIdx, headIdx, :, :] using direct memory copy. |
| `GetAdjacencyMatrix` |  |
| `GetAdjacencySlice(Tensor<>,Int32,Boolean)` | Gets the adjacency matrix slice for a batch (handles both 2D and 3D cases). |
| `GetAdjacencyValue(Int32,Int32,Int32)` | Helper to get adjacency value - supports both 2D [nodes, nodes] and 3D [batch, nodes, nodes]. |
| `GetMetadata` | Returns layer-specific metadata for serialization (numHeads, alpha, dropoutRate). |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes layer parameters using Xavier/Glorot initialization with Engine operations. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `Set2DSliceIn4D(Tensor<>,Int32,Int32,Tensor<>)` | Sets a 2D slice in a 4D tensor at position [batchIdx, headIdx, :, :] using direct memory copy. |
| `Set3DSliceIn4DForHead(Tensor<>,Int32,Tensor<>)` | Sets a 3D slice [h, :, :] into a 4D tensor at [b, h, :, :] for all batches. |
| `SetAdjacencyMatrix(Tensor<>)` |  |
| `SetEdges(Tensor<Int32>,Tensor<Int32>)` | Sets the edge list representation of the graph structure for sparse aggregation. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adjacencyMatrix` | The adjacency matrix defining graph structure. |
| `_attentionWeights` | Attention mechanism parameters tensor. |
| `_attentionWeightsGradient` | Gradients for attention parameters. |
| `_bias` | Bias tensor for the output transformation. |
| `_biasGradient` | Gradients for bias parameters. |
| `_edgeSourceIndices` | Edge source node indices for sparse graph representation. |
| `_edgeTargetIndices` | Edge target node indices for sparse graph representation. |
| `_lastAttentionCoefficients` | Cached attention coefficients from forward pass. |
| `_lastHeadOutputs` | Cached head outputs before averaging. |
| `_lastInput` | Cached input from forward pass for backward computation. |
| `_lastOutput` | Cached output from forward pass for backward computation. |
| `_lastPreSoftmaxScores` | Cached pre-softmax attention scores for gradient computation. |
| `_lastTransformed` | Cached transformed features from forward pass for gradient computation. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_useSparseAggregation` | Indicates whether to use sparse (edge-based) or dense (adjacency matrix) aggregation. |
| `_weights` | Weight tensor for each attention head. |
| `_weightsGradient` | Gradients for weight parameters. |

