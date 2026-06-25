---
title: "EdgeConditionalConvolutionalLayer<T>"
description: "Implements Edge-Conditioned Convolution for incorporating edge features in graph convolutions."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements Edge-Conditioned Convolution for incorporating edge features in graph convolutions.

## For Beginners

This layer lets connections (edges) have their own properties.

Think of a transportation network:

- Regular graph layers: All roads are treated the same
- Edge-conditioned layers: Each road has properties (speed limit, distance, traffic)

Examples where edge features matter:

- **Molecules**: Bond types (single/double/triple) affect how atoms interact
- **Social networks**: Relationship types (friend/colleague/family) influence information flow
- **Knowledge graphs**: Relationship types (is-a/part-of/located-in) determine connections
- **Transportation**: Road types (highway/street/path) affect travel patterns

This layer learns how to use these edge properties to better aggregate neighbor information.

## How It Works

Edge-Conditioned Convolutions extend standard graph convolutions by incorporating edge features
into the aggregation process. Instead of treating all edges equally, this layer learns
edge-specific transformations based on edge attributes.

The layer computes: h_i' = σ(Σ_{j∈N(i)} θ(e_ij) · h_j + b)
where θ(e_ij) is an edge-specific transformation learned from edge features e_ij.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EdgeConditionalConvolutionalLayer(Int32,Int32,Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `EdgeConditionalConvolutionalLayer` class. |

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
| `ApplyReLU(Tensor<>)` | Applies ReLU activation element-wise to a tensor. |
| `BatchedMatMul3Dx2D(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32)` | Performs batched matrix multiplication between a 3D tensor and a 2D weight matrix. |
| `BroadcastBias(Tensor<>,Int32,Int32)` | Broadcasts a bias tensor across batch and node dimensions. |
| `BuildEdgeIndicesFromAdjacency(Tensor<>,Int32,Int32,Int32)` | Builds source and target node indices from adjacency matrix for GPU edge processing. |
| `ComputeEdgeConditionedAggregationCpu(Single[],Single[],Tensor<>,Int32,Int32,Int32,Int32,Int32)` | Computes edge-conditioned aggregation on CPU due to edge-specific indexing. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` |  |
| `GetAdjacencyMatrix` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `NormalizeAdjacency(Tensor<>,Int32,Int32)` | Normalizes adjacency matrix by degree (row normalization) and ensures batch dimension. |
| `NormalizeEdgeFeatures(Tensor<>,Int32)` | Normalizes edge features to include batch dimension. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetAdjacencyMatrix(Tensor<>)` |  |
| `SetEdgeFeatures(Tensor<>)` | Sets the edge features for this layer. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adjacencyMatrix` | The adjacency matrix defining graph structure. |
| `_bias` | Bias vector. |
| `_edgeFeatures` | Edge features tensor. |
| `_edgeNetworkWeights1` | Edge network: transforms edge features to weight matrices. |
| `_edgeNetworkWeights1Gradient` | Gradients. |
| `_lastInput` | Cached values for backward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_selfWeights` | Self-loop transformation weights. |

