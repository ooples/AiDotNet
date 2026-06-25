---
title: "DirectionalGraphLayer<T>"
description: "Implements Directional Graph Networks for directed graph processing with separate in/out aggregations."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements Directional Graph Networks for directed graph processing with separate in/out aggregations.

## For Beginners

This layer understands that graph connections can have direction.

Think of different types of directed networks:

**Twitter/Social Media:**

- You follow someone (outgoing edge)
- Someone follows you (incoming edge)
- These are NOT the same! Celebrities have many incoming, fewer outgoing

**Citation Networks:**

- Papers you cite (outgoing): Shows your influences
- Papers citing you (incoming): Shows your impact
- Direction matters for understanding importance

**Web Pages:**

- Links you have (outgoing): What you reference
- Links to you (incoming/backlinks): Your authority
- Google PageRank uses this directionality

**Transaction Networks:**

- Money sent (outgoing): Your purchases
- Money received (incoming): Your sales
- Different patterns for buyers vs sellers

Why separate in/out aggregation?

- **Asymmetric roles**: Being cited vs citing have different meanings
- **Different patterns**: Incoming and outgoing patterns can be very different
- **Better expressiveness**: Captures more information than treating edges as undirected

The layer learns separate transformations for incoming and outgoing neighbors,
then combines them to update each node's representation.

## How It Works

Directional Graph Networks (DGN) explicitly model the directionality of edges in directed graphs.
Unlike standard GNNs that often ignore edge direction or treat graphs as undirected, DGNs
maintain separate aggregations for incoming and outgoing edges, capturing asymmetric relationships.

The layer computes separate representations for in-neighbors and out-neighbors:

- h_in = AGGREGATE_IN({h_j : j → i})
- h_out = AGGREGATE_OUT({h_j : i → j})
- h_i' = UPDATE(h_i, h_in, h_out)

This allows the network to learn different patterns for sources and targets of edges.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DirectionalGraphLayer(Int32,Int32,Boolean,IActivationFunction<>)` | Initializes a new instance of the `DirectionalGraphLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputFeatures` |  |
| `OutputFeatures` |  |
| `ParameterCount` |  |
| `SupportsGpuExecution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGatesToFeatures(Tensor<>,Tensor<>,Int32,Int32,Int32)` | Applies gates to the concatenated features. |
| `ApplyGatesToFeaturesGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,IGpuBuffer,Int32,Int32)` | Applies learned gates to feature groups on GPU. |
| `ApplySigmoid(Tensor<>)` | Applies sigmoid activation element-wise to a tensor. |
| `BackwardThroughGating(Tensor<>,Int32,Int32)` | Computes gradients through the gating mechanism. |
| `BatchedMatMul3Dx2D(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32)` | Performs batched matrix multiplication between a 3D tensor and a 2D weight matrix. |
| `BroadcastBias(Tensor<>,Int32,Int32)` | Broadcasts a bias tensor across batch and node dimensions. |
| `ClearGradients` |  |
| `ConcatenateFeatures(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Concatenates incoming, outgoing, and self features along the feature dimension. |
| `ConvertToCSR(Tensor<>)` | Converts a dense adjacency matrix to CSR format. |
| `ConvertToCSRTranspose(Tensor<>)` | Converts a dense adjacency matrix to CSR format of its transpose. |
| `CopyToOutputBuffer(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32,Int32)` | Copies data from source buffer to a specific offset in the destination buffer. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass for DirectionalGraphLayer. |
| `GetAdjacencyMatrix` |  |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetAdjacencyMatrix(Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adjForBatch` | The adjacency matrix reshaped to 3D for batched operations. |
| `_adjacencyMatrix` | The adjacency matrix defining graph structure (interpreted as directed). |
| `_combinationBias` | Combination bias. |
| `_combinationWeights` | Combination weights for merging in/out/self features. |
| `_gateBias` | Gating bias (if enabled). |
| `_gateWeights` | Gating mechanism weights (if enabled). |
| `_incomingBias` | Biases for incoming, outgoing, and self transformations. |
| `_incomingWeights` | Weights for incoming edge aggregation. |
| `_incomingWeightsGradient` | Gradients. |
| `_lastInput` | Cached values for backward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_outgoingWeights` | Weights for outgoing edge aggregation. |
| `_selfWeights` | Self-loop weights. |

