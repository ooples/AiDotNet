---
title: "BatchEnsembleLayer<T>"
description: "Implements a BatchEnsemble layer that provides parameter-efficient ensembling."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a BatchEnsemble layer that provides parameter-efficient ensembling.

## For Beginners

BatchEnsemble is a clever way to create multiple models
(ensemble members) that share most of their weights.

Traditional ensemble: Train N separate models with N×parameters
BatchEnsemble: Train 1 base model + N small vectors = ~1×parameters + small overhead

How it works:

1. A single shared weight matrix W captures the main learned patterns
2. Each ensemble member has two small vectors (r and s)
3. Member i's effective weights = W × (r_i outer-product s_i)
4. This modulates the shared weights to create diversity

Benefits:

- Ensemble predictions (averaging multiple members) are usually more accurate
- Parameter cost is only slightly more than a single model
- Can process all members in parallel using batch operations
- Easy to implement and train

Example with 256-dim hidden layer and 4 members:

- Shared weights: 256 × 256 = 65,536 parameters
- Per-member vectors: 4 × (256 + 256) = 2,048 parameters
- Total overhead: ~3% more parameters for 4× ensemble benefit

## How It Works

BatchEnsemble creates multiple ensemble members that share base weights but have
their own small rank-1 matrices. For a weight matrix W, each member i computes:
W_i = W ⊙ (r_i ⊗ s_i)
where r_i and s_i are per-member rank vectors, ⊙ is element-wise multiplication,
and ⊗ is outer product.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BatchEnsembleLayer(Int32,Int32,Int32,Boolean,Double)` | Initializes a new instance of the BatchEnsembleLayer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputDim` | Gets the input dimension. |
| `NumMembers` | Gets the number of ensemble members. |
| `OutputDim` | Gets the output dimension. |
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AverageMembers(Tensor<>)` | Averages the outputs across ensemble members. |
| `Forward(Tensor<>)` | Performs the forward pass through the BatchEnsemble layer. |
| `ForwardExpanded(Tensor<>)` | Forward pass for input that is ALREADY expanded along the member axis (`[batchSize * numMembers, inputDim]`, members in consecutive rows). |
| `GetParameterGradients` | Gets the parameter gradients as a single vector. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetRVectors` | Gets the r vectors tensor. |
| `GetSVectors` | Gets the s vectors tensor. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWeights` | Gets the shared weights tensor. |
| `InitializeRankVectors(Tensor<>,Double)` | Initializes rank vectors centered around 1 with some variation. |
| `InitializeXavier(Tensor<>)` | Initializes a tensor with Xavier/Glorot initialization. |
| `ResetGradients` | Resets all gradients. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters from a vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

