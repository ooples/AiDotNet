---
title: "MeshPoolLayer<T>"
description: "Implements mesh pooling via edge collapse for MeshCNN-style networks."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements mesh pooling via edge collapse for MeshCNN-style networks.

## For Beginners

Just like max pooling shrinks an image by combining pixels,
mesh pooling shrinks a mesh by combining edges. The layer learns which edges are
less important and removes them, simplifying the mesh while preserving important features.

Key concepts:

- Edge collapse: Remove an edge by merging its two vertices into one
- Importance score: Learned value indicating how important each edge is
- Target edges: Number of edges to keep after pooling

The process:

1. Compute importance scores for all edges using current features
2. Sort edges by importance (lowest first)
3. Collapse least important edges until target count is reached
4. Update adjacency information for remaining edges

## How It Works

MeshPoolLayer reduces the number of edges in a mesh by collapsing edges based on
learned importance scores. This is analogous to pooling in image CNNs but operates
on the mesh structure.

Reference: "MeshCNN: A Network with an Edge" by Hanocka et al., SIGGRAPH 2019

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeshPoolLayer(Int32,Int32,Int32)` | Initializes a new instance of the `MeshPoolLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputChannels` | Gets the number of input feature channels per edge. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `RemainingEdgeIndices` | Gets or sets the edge indices that remain after pooling. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |
| `TargetEdges` | Gets the target number of edges after pooling. |
| `UpdatedAdjacency` | Gets or sets the updated edge adjacency after pooling. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of the layer. |
| `ComputeImportanceScores(Tensor<>)` | Computes importance scores for all edges using vectorized matrix-vector multiplication. |
| `ComputeImportanceWeightsGradient(Tensor<>,Tensor<>,Int32[])` | Computes gradient for importance weights using vectorized operations. |
| `Deserialize(BinaryReader)` | Deserializes the layer from a binary stream. |
| `Forward(Tensor<>)` | Performs the forward pass of mesh pooling. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass for mesh pooling. |
| `GetBiases` | Gets the bias tensor (null for this layer). |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWeights` | Gets the importance weights tensor. |
| `InitializeWeights` | Initializes importance weights with small random values. |
| `ResetState` | Resets the cached state from forward/backward passes. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `Serialize(BinaryWriter)` | Serializes the layer to a binary stream. |
| `SetEdgeAdjacency(Int32[0:,0:])` | Sets the edge adjacency information for the current mesh. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SortEdgesByImportance(Tensor<>,Int32)` | Sorts edge indices by their importance scores. |
| `UpdateAdjacency(Int32[0:,0:],Int32[],Int32)` | Updates edge adjacency after removing edges. |
| `UpdateParameters()` | Updates the layer parameters using computed gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuInput` | Cached GPU input for backward pass. |
| `_gpuInputShape` | Cached GPU input shape for backward pass. |
| `_importanceWeights` | Learnable weights for computing edge importance scores. |
| `_importanceWeightsGradient` | Cached gradient for importance weights. |
| `_lastEdgeAdjacency` | Cached edge adjacency from the last forward pass. |
| `_lastImportanceScores` | Cached importance scores from the last forward pass. |
| `_lastInput` | Cached input from the last forward pass. |
| `_lastOutput` | Cached output from the last forward pass. |
| `_numNeighbors` | Number of neighboring edges per edge (default 4 for triangular meshes). |

