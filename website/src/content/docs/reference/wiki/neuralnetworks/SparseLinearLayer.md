---
title: "SparseLinearLayer<T>"
description: "Represents a fully connected layer with sparse weight matrix for efficient computation."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a fully connected layer with sparse weight matrix for efficient computation.

## For Beginners

This layer works like a regular dense layer, but uses
sparse matrices to store weights more efficiently.

Benefits of sparse layers:

- Much less memory for large layers with few active connections
- Faster computation (only non-zero weights are used)
- Natural for network pruning and compression

Use cases:

- Graph neural networks (sparse adjacency)
- Recommender systems (sparse user-item matrices)
- Pruned neural networks
- Very large embedding layers

## How It Works

A sparse linear layer is similar to a dense layer but uses sparse weight storage.
This is efficient when most weights are zero (or can be pruned), reducing both
memory usage and computation time.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SparseLinearLayer(Int32,Int32,Double,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the SparseLinearLayer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputFeatures` | Gets the number of input features. |
| `OutputFeatures` | Gets the number of output features. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` | Gets whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` | Resets the internal state of the layer. |
| `ComputeGradients(Tensor<>)` | Computes gradients with respect to weights, biases, and input. |
| `Forward(Tensor<>)` | Performs the forward pass through the layer. |
| `GetMetadata` | Adds sparsity to the layer metadata so deserialization rebuilds the layer with the same sparsity ratio and activation function. |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeSparseWeights` | Initializes sparse weights using the layer's initialization strategy. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `ShapeMatchesLastOutput(Tensor<>)` | Returns true when `grad` shape is consistent with the last forward's output. |
| `TransposeMatrix(Matrix<>)` | Transposes a matrix using O(1) stride-based view. |
| `UpdateParameters()` | Updates the layer's parameters using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | The bias values, registered as a tape-trainable parameter alongside `_weights`. |
| `_biasesGradient` | Gradient for biases, stored during backward pass. |
| `_lastInput` | Stored input from forward pass for backpropagation. |
| `_lastOutput` | Stored pre-activation output for gradient computation. |
| `_sparsity` | The sparsity level (fraction of weights that are zero). |
| `_weights` | The sparse weight matrix. |
| `_weightsGradient` | Gradient for weights, stored during backward pass. |

