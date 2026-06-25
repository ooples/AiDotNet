---
title: "GraphIsomorphismLayer<T>"
description: "Implements Graph Isomorphism Network (GIN) layer for powerful graph representation learning."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements Graph Isomorphism Network (GIN) layer for powerful graph representation learning.

## How It Works

Graph Isomorphism Networks (GIN), introduced by Xu et al., are provably as powerful as the
Weisfeiler-Lehman (WL) graph isomorphism test for distinguishing graph structures. GIN uses
a sum aggregation with a learnable epsilon parameter and applies a multi-layer perceptron (MLP)
to the aggregated features.

The layer computes: h_v^(k) = MLP^(k)((1 + epsilon^(k)) * h_v^(k-1) + sum_{u in N(v)} h_u^(k-1))
where h_v is the representation of node v, N(v) is the neighborhood of v,
epsilon is a learnable parameter (or fixed), and MLP is a multi-layer perceptron.

**Production-Ready Features:**

- Fully vectorized operations using IEngine for GPU acceleration
- Tensor-based weights for all parameters
- Dual backward pass: BackwardManual() for efficiency, BackwardViaAutodiff() for accuracy
- Full gradient computation through MLP and aggregation paths
- Complete GetParameters()/SetParameters() for model persistence

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphIsomorphismLayer(Int32,Int32,Int32,Boolean,Double,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `GraphIsomorphismLayer` class. |

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
| `BatchedMatMul3Dx2D(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32)` | Performs batched matrix multiplication between a 3D tensor and a 2D weight matrix. |
| `BroadcastBias(Tensor<>,Int32,Int32)` | Broadcasts a bias tensor across batch and node dimensions using Engine operations. |
| `ComputeGradientsViaEngine(Tensor<>,Int32,Int32)` | Computes gradients using fully vectorized Engine operations. |
| `CopyToOffsetCpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32,Int32)` | CPU fallback for copying data to an offset in a destination buffer. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass for Graph Isomorphism Network (GIN). |
| `GetAdjacencyMatrix` |  |
| `GetMetadata` | Returns layer-specific metadata for serialization. |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes layer parameters using Xavier initialization with Engine operations. |
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
| `_adjacencyMatrix` | The adjacency matrix defining graph structure. |
| `_epsilon` | Epsilon parameter for weighting self vs neighbor features. |
| `_epsilonGradient` | Gradients for epsilon. |
| `_lastAggregated` | Cached aggregated features (before MLP). |
| `_lastInput` | Cached input from forward pass. |
| `_lastMlpHidden` | Cached hidden layer output from MLP (after ReLU). |
| `_lastMlpHiddenPreRelu` | Cached pre-ReLU hidden layer output from MLP. |
| `_lastNeighborSum` | Cached neighbor sum before applying epsilon. |
| `_lastOutput` | Cached output from forward pass. |
| `_mlpBias1` | Bias for first MLP layer: [mlpHiddenDim]. |
| `_mlpBias2` | Bias for second MLP layer: [outputFeatures]. |
| `_mlpWeights1` | First layer of the MLP: [inputFeatures, mlpHiddenDim]. |
| `_mlpWeights1Gradient` | Gradients for MLP weights. |
| `_mlpWeights2` | Second layer of the MLP: [mlpHiddenDim, outputFeatures]. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

