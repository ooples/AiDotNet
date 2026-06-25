---
title: "PrincipalNeighbourhoodAggregationLayer<T>"
description: "Implements Principal Neighbourhood Aggregation (PNA) layer for powerful graph representation learning."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements Principal Neighbourhood Aggregation (PNA) layer for powerful graph representation learning.

## How It Works

Principal Neighbourhood Aggregation (PNA), introduced by Corso et al., addresses limitations
of existing GNN architectures by using multiple aggregators and scalers. PNA combines:

1. Multiple aggregation functions (mean, max, min, sum, std)
2. Multiple scaling functions to normalize by degree
3. Learnable combination of all aggregated features

The layer computes: h_i' = MLP(COMBINE({SCALE(AGGREGATE({h_j : j in N(i)}))}))
where AGGREGATE in {mean, max, min, sum, std}, SCALE in {identity, amplification, attenuation},
and COMBINE is a learned linear combination followed by MLP.

**Production-Ready Features:**

- Fully vectorized operations using IEngine for GPU acceleration
- BatchMatMul for efficient batched graph operations
- Dual backward pass: BackwardManual() for efficiency, BackwardViaAutodiff() for accuracy
- Full gradient computation through all aggregators and scalers
- Complete GetParameters()/SetParameters() for model persistence

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrincipalNeighbourhoodAggregationLayer(Int32,Int32,PNAAggregator[],PNAScaler[],Double,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `PrincipalNeighbourhoodAggregationLayer` class. |

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
| `ApplyFusedActivation(IDirectGpuBackend,IGpuBuffer,Int32)` | Applies the fused activation function on GPU. |
| `ApplyScalerGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,IGpuBuffer,Int32,Int32,Single,Boolean)` | Applies amplification or attenuation scaler on GPU. |
| `ApplyVectorizedScaler(Tensor<>,PNAScaler,Tensor<>)` | Applies scaler to aggregated features using vectorized operations. |
| `BackpropThroughAggregation(Tensor<>,Int32)` | Backpropagates through aggregation operations using vectorized Engine operations. |
| `BroadcastBias(Tensor<>,Int32,Int32)` | Broadcasts a bias tensor across batch and node dimensions using Engine operations. |
| `ClearGradients` |  |
| `ComputeDegrees(Tensor<>,Int32)` | Computes node degrees from adjacency matrix. |
| `ComputeGradientsViaEngine(Tensor<>,Int32,Int32)` | Computes gradients using vectorized Engine operations as fallback for autodiff. |
| `ComputeMaxAggregation(Tensor<>,Int32)` | Computes max aggregation over neighbors using masked reduce. |
| `ComputeMinAggregation(Tensor<>,Int32)` | Computes min aggregation over neighbors using masked reduce. |
| `ComputeStdDevAggregation(Tensor<>,Tensor<>,Int32)` | Computes standard deviation aggregation using vectorized operations. |
| `ComputeVectorizedAggregation(Tensor<>,PNAAggregator,Tensor<>,Int32)` | Computes vectorized aggregation using Engine operations. |
| `ConvertToCSR(Tensor<>,Int32)` | Converts adjacency matrix to CSR format. |
| `CopyToConcat(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32,Int32,Int32)` | Copies scaled features to the concatenated buffer at the specified offset using GPU strided copy. |
| `DivideByDegreeGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32,Int32)` | Divides each feature row by the corresponding node degree on GPU. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass for PrincipalNeighbourhoodAggregationLayer. |
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
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

