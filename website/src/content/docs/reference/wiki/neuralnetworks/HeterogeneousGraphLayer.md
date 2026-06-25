---
title: "HeterogeneousGraphLayer<T>"
description: "Implements Heterogeneous Graph Neural Network layer for graphs with multiple node and edge types."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements Heterogeneous Graph Neural Network layer for graphs with multiple node and edge types.

## For Beginners

This layer handles graphs where not all nodes and edges are the same.

Real-world examples:

**Knowledge Graph:**

- Node types: Person, Place, Event
- Edge types: BornIn, HappenedAt, AttendedBy
- Each type needs different processing

**E-commerce:**

- Node types: User, Product, Brand, Category
- Edge types: Purchased, Manufactured, BelongsTo, Viewed
- Different relationships have different meanings

**Academic Network:**

- Node types: Author, Paper, Venue, Topic
- Edge types: Wrote, PublishedIn, About, Cites
- Mixed types of entities and relationships

Why heterogeneous?

- **Different semantics**: A "User" has different properties than a "Product"
- **Type-specific patterns**: Relationships mean different things
- **Better representation**: Specialized processing for each type

The layer learns separate transformations for each edge type, then combines them intelligently.

## How It Works

Heterogeneous Graph Neural Networks (HGNNs) handle graphs where nodes and edges have different types.
Unlike homogeneous GNNs that treat all nodes and edges uniformly, HGNNs use type-specific
transformations and aggregations. This layer implements the R-GCN (Relational GCN) approach
with type-specific weight matrices.

The layer computes: h_i' = σ(Σ_{r∈R} Σ_{j∈N_r(i)} (1/c_{i,r}) W_r h_j + W_0 h_i)
where R is the set of relation types, N_r(i) are neighbors of type r, c_{i,r} is a normalization
constant, W_r are relation-specific weights, and W_0 is the self-loop weight.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HeterogeneousGraphLayer(HeterogeneousGraphMetadata,Int32,Boolean,Int32,IActivationFunction<>)` | Initializes a new instance of the `HeterogeneousGraphLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputFeatures` |  |
| `OutputFeatures` |  |
| `ParameterCount` |  |
| `SupportsGpuExecution` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNodeOutput(Tensor<>,Tensor<>,Tensor<>,Int32,Int32,Int32)` | Adds node output and bias to the overall output tensor. |
| `BatchedMatMul3Dx2D(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32)` | Performs batched matrix multiplication between a 3D tensor and a 2D weight matrix. |
| `BroadcastBias(Tensor<>,Int32)` | Broadcasts a bias tensor across batch dimension. |
| `ConvertToCSR(Tensor<>,Int32)` | Converts a normalized adjacency tensor to CSR format (using first batch element). |
| `ExtractBasisMatrix(Tensor<>,Int32,Int32,Int32)` | Extracts a basis matrix from the basis matrices tensor. |
| `ExtractBatchSlice(Tensor<>,Int32,Int32,Int32)` | Extracts a 2D batch slice from a 3D tensor. |
| `ExtractInputFeatures(Tensor<>,Int32,Int32,Int32)` | Extracts input features for a specific number of features. |
| `ExtractNodeInput(Tensor<>,Int32,Int32,Int32)` | Extracts input for a specific node across all batches. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass for HeterogeneousGraphLayer. |
| `GetAdjacencyMatrix` |  |
| `GetParameterTensors` | Gets all trainable parameters of the layer as a list of tensors. |
| `GetParameters` |  |
| `InitializeTensor(Tensor<>,)` | Initializes a tensor with scaled random values. |
| `NormalizeAdjacency(Tensor<>,Int32,Int32)` | Normalizes adjacency matrix by degree (row normalization). |
| `ResetState` |  |
| `SetAdjacencyMatrices(Dictionary<String,Tensor<>>)` | Sets the adjacency matrices for all edge types. |
| `SetAdjacencyMatrix(Tensor<>)` |  |
| `SetNodeTypeMap(Dictionary<Int32,String>)` | Sets the node type mapping. |
| `SetParameterTensors(List<Tensor<>>)` | Sets the trainable parameters of the layer from a list of tensors. |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` | Updates the layer parameters based on computed gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adjacencyMatrices` | The adjacency matrices for each edge type. |
| `_basisCoefficients` | Coefficients for combining basis matrices per edge type. |
| `_basisMatrices` | Basis matrices for weight decomposition (if using basis). |
| `_biases` | Bias for each node type. |
| `_edgeTypeWeights` | Type-specific weight tensors. |
| `_edgeTypeWeightsGradients` | Gradients for weights, self-loop weights, and biases. |
| `_lastInput` | Cached values for backward pass. |
| `_nodeTypeMap` | Node type assignments for each node. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_selfLoopWeights` | Self-loop weights for each node type. |

