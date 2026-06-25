---
title: "GraphSAGELayer<T>"
description: "Implements GraphSAGE (Graph Sample and Aggregate) layer for inductive learning on graphs."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements GraphSAGE (Graph Sample and Aggregate) layer for inductive learning on graphs.

## How It Works

GraphSAGE, introduced by Hamilton et al., is designed for inductive learning on graphs,
meaning it can generalize to unseen nodes and graphs. Instead of learning embeddings for
each node directly, it learns aggregator functions that generate embeddings by sampling
and aggregating features from a node's local neighborhood.

The layer performs: h_v = sigma(W_self * h_v + W_neigh * AGGREGATE({h_u : u in N(v)}) + b)
where h_v is the representation of node v, N(v) is the neighborhood of v,
AGGREGATE is an aggregation function (mean, max, sum), and sigma is an activation function.

**Production-Ready Features:**

- Fully vectorized operations using IEngine for GPU acceleration
- Tensor-based weights for all parameters
- Dual backward pass: BackwardManual() for efficiency, BackwardViaAutodiff() for accuracy
- Full gradient computation through aggregation paths
- Complete GetParameters()/SetParameters() for model persistence

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphSAGELayer(Int32,Int32,SAGEAggregatorType,Boolean,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `GraphSAGELayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputFeatures` |  |
| `OutputFeatures` |  |
| `ParameterCount` |  |
| `SupportsGpuExecution` | Gets whether this layer supports GPU execution. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivationCpu(IDirectGpuBackend,IGpuBuffer,Int32,FusedActivationType)` | CPU fallback for applying activation function. |
| `BackpropThroughAggregation(Tensor<>,Int32,Int32)` | Backpropagates through aggregation operation. |
| `BackpropThroughL2Norm(Tensor<>,Tensor<>,Int32,Int32)` | Backpropagates through L2 normalization. |
| `BatchedMatMul3Dx2D(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32)` | Performs batched matrix multiplication between a 3D tensor and a 2D weight matrix. |
| `BroadcastBias(Tensor<>,Int32,Int32)` | Broadcasts a bias tensor across batch and node dimensions using Engine operations. |
| `ComputeGradientsViaEngine(Tensor<>,Int32,Int32)` | Computes gradients using fully vectorized Engine operations as fallback. |
| `ComputeMaxAggregation(Tensor<>,Int32,Int32,Tensor<>)` | Computes max aggregation over neighbors using masked reduce. |
| `ComputeVectorizedAggregation(Tensor<>,Tensor<>,Int32,Int32,Tensor<>)` | Computes vectorized aggregation using Engine operations. |
| `CopyToOffsetCpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32,Int32)` | CPU fallback for copying data to an offset in a destination buffer. |
| `DivideByRowDegreeCpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32,Int32)` | CPU fallback for dividing each row by its degree. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass for GraphSAGE layer. |
| `GetAdjacencyMatrix` |  |
| `GetMetadata` | Returns layer-specific metadata for serialization (aggregatorType, normalize). |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes layer parameters using Xavier initialization with Engine operations. |
| `L2NormalizeRowsCpu(IDirectGpuBackend,IGpuBuffer,Int32,Int32)` | CPU fallback for L2 row normalization. |
| `L2NormalizeVectorized(Tensor<>,Int32,Int32)` | Applies L2 normalization using vectorized Engine operations. |
| `MaxPoolNeighborsCpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,IGpuBuffer,Int32,Int32)` | CPU fallback for max pooling over neighbors. |
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
| `_adjForBatch` | Cached reshaped adjacency matrix for backward pass. |
| `_adjacencyMatrix` | The adjacency matrix defining graph structure. |
| `_bias` | Bias tensor. |
| `_biasGradient` | Gradients for bias. |
| `_lastAggregated` | Cached aggregated neighbor features. |
| `_lastDegrees` | Cached degrees for each node. |
| `_lastInput` | Cached input from forward pass. |
| `_lastMaxIndices` | Cached max indices from MaxPool aggregation for proper backward pass. |
| `_lastMaxInputShape` | Cached input shape before max pooling for backward computation. |
| `_lastOutput` | Cached output from forward pass. |
| `_lastPreNorm` | Cached pre-normalization output for gradient computation. |
| `_neighborWeights` | Weight tensor for neighbor features. |
| `_neighborWeightsGradient` | Gradients for neighbor weights. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_selfWeights` | Weight tensor for self features. |
| `_selfWeightsGradient` | Gradients for self weights. |

