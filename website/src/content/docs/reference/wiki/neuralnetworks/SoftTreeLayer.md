---
title: "SoftTreeLayer<T>"
description: "A differentiable soft decision tree layer for GANDALF and similar architectures."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A differentiable soft decision tree layer for GANDALF and similar architectures.

## For Beginners

A soft tree is like a fuzzy decision tree:

- Regular tree: "Is age > 30? Go left or right"
- Soft tree: "Is age > 30? Go 70% left, 30% right"

The soft splits make the tree trainable with neural network methods while maintaining
the interpretable structure of decision trees.

## How It Works

This layer implements a soft (differentiable) decision tree that can be trained with gradient descent.
Each internal node uses soft splits (sigmoid) instead of hard decisions, allowing gradients to flow
through the tree structure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SoftTreeLayer(Int32,Int32,Int32,Double,Double)` | Initializes a new soft tree layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Depth` | Gets the tree depth. |
| `NumLeaves` | Gets the number of leaf nodes in this tree. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePathProbabilities(Tensor<>,Int32)` | Computes the probability of reaching each leaf node. |
| `Deserialize(BinaryReader)` |  |
| `Forward(Tensor<>)` | Forward pass through the soft tree. |
| `GetFeatureImportance` | Gets feature importance based on split weight magnitudes. |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `Serialize(BinaryWriter)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `UpdateParameters(Vector<>)` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedRightProbs` |  |

