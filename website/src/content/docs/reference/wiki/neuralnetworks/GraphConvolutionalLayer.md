---
title: "GraphConvolutionalLayer<T>"
description: "Represents a Graph Convolutional Network (GCN) layer for processing graph-structured data."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Graph Convolutional Network (GCN) layer for processing graph-structured data.

## For Beginners

This layer helps neural networks understand data that's organized like a network or graph.

Think of a social network where people are connected to friends:

- Each person is a "node" with certain features (age, interests, etc.)
- Connections between people are "edges"
- This layer helps the network learn patterns by looking at each person AND their connections

For example, in a social network recommendation system, this layer can help understand that 
a person might like something because their friends like it, even if their personal profile 
doesn't suggest they would.

## How It Works

A Graph Convolutional Layer applies convolution operations to graph-structured data by leveraging
an adjacency matrix that defines connections between nodes in the graph. This layer learns
representations for nodes in a graph by aggregating feature information from a node's local neighborhood.
The layer performs the transformation: output = adjacency_matrix * input * weights + bias.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphConvolutionalLayer(Int32,Int32,IActivationFunction<>,Boolean)` | Initializes a new instance of the `GraphConvolutionalLayer` class with the specified dimensions and activation function. |
| `GraphConvolutionalLayer(Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `GraphConvolutionalLayer` class with the specified dimensions and vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the auxiliary loss contribution. |
| `InputFeatures` | Gets the number of input features per node. |
| `OutputFeatures` | Gets the number of output features per node. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SmoothnessWeight` | Gets or sets the weight for Laplacian smoothness regularization. |
| `SupportsGpuExecution` | Gets whether this layer has GPU execution support. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |
| `UseAuxiliaryLoss` | Gets or sets a value indicating whether auxiliary loss is enabled for this layer. |
| `UsesSparseAggregation` | Gets whether sparse (edge-based) aggregation is currently enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BatchedMatMul3Dx2D(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32)` | Performs batched matrix multiplication between a 3D tensor and a 2D weight matrix. |
| `BroadcastBias(Tensor<>,Int32,Int32)` | Broadcasts a bias tensor across batch and node dimensions. |
| `ClearEdges` | Clears the edge list and switches back to dense adjacency matrix aggregation. |
| `ClearGradients` | Clears the stored gradients for this layer. |
| `ComputeAuxiliaryLoss` | Computes the Laplacian smoothness regularization loss on node features. |
| `EnableImplicitIdentityAdjacency` | Sets the adjacency matrix that defines the graph structure. |
| `EnsureDenseAdjacencyForInput(Tensor<>)` | Performs the forward pass of the graph convolutional layer. |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass for graph convolution using sparse matrix operations. |
| `GetAdjacencyMatrix` | Gets the adjacency matrix currently being used by this layer. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary loss computation. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetParameterGradients` | Gets the gradients of all trainable parameters in this layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the weights and biases of the layer. |
| `InitializeTensor(Tensor<>,)` | Initializes a tensor with scaled random values. |
| `ResetState` | Resets the internal state of the layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetEdges(Tensor<Int32>,Tensor<Int32>)` | Sets the edge list representation of the graph structure for sparse aggregation. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adjForBatch` | Cached reshaped adjacency matrix (3D) for backward pass. |
| `_adjacencyMatrix` | The adjacency matrix that defines the graph structure. |
| `_bias` | The bias tensor that is added to the output of the transformation. |
| `_biasGradient` | Stores the gradients for the bias calculated during the backward pass. |
| `_edgeSourceIndices` | Edge source node indices for sparse graph representation. |
| `_edgeTargetIndices` | Edge target node indices for sparse graph representation. |
| `_edgesExtracted` | Tracks whether edges have been extracted from the current adjacency matrix. |
| `_graphEdges` | Stores the list of edges in the graph for auxiliary loss computation. |
| `_lastGraphSmoothnessLoss` | Stores the last computed graph smoothness loss for diagnostic purposes. |
| `_lastInput` | Stores the input tensor from the last forward pass for use in the backward pass. |
| `_lastNodeFeatures` | Stores the node features from the last forward pass for auxiliary loss computation. |
| `_lastOutput` | Stores the output tensor from the last forward pass for use in the backward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_useSparseAggregation` | Indicates whether to use sparse (edge-based) or dense (adjacency matrix) aggregation. |
| `_weights` | The weight tensor that transforms input features to output features. |
| `_weightsGradient` | Stores the gradients for the weights calculated during the backward pass. |

