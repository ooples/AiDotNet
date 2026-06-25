---
title: "ObliviousDecisionTreeLayer<T>"
description: "Oblivious Decision Tree (ODT) for NODE architecture."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Oblivious Decision Tree (ODT) for NODE architecture.

## For Beginners

An oblivious tree is a special type of decision tree where:

- At level 1, ALL nodes use the same feature (e.g., "age > 30")
- At level 2, ALL nodes use the same feature (e.g., "income > 50k")
- And so on...

This is simpler than regular trees where each node can use different features.
The simplicity helps prevent overfitting and makes the tree faster.

## How It Works

An oblivious decision tree uses the same feature and threshold at each level,
making it more regularized and efficient than standard decision trees.
NODE uses differentiable ODTs with entmax splits for end-to-end learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ObliviousDecisionTreeLayer(Int32,Int32,Double)` | Lazy constructor: resolves `inputDim` from `input.Shape[^1]` on first `Tensor{`. |
| `ObliviousDecisionTreeLayer(Int32,Int32,Int32,Double)` | Initializes an oblivious decision tree. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumLeaves` | Gets the number of leaf nodes (2^depth). |
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsureInitialized` |  |
| `Forward(Tensor<>)` | Forward pass through the oblivious decision tree. |
| `GetFeatureImportance` | Gets feature importance based on selection weights. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

