---
title: "InteractingLayer<T>"
description: "Interacting Layer for AutoInt architecture."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Interacting Layer for AutoInt architecture.

## For Beginners

The interacting layer helps discover relationships between features:

- 1st layer: "age relates to income"
- 2nd layer: "age + income together relate to credit score"
- 3rd layer: "age + income + credit score relate to loan approval"

Each layer builds on the previous to capture more complex patterns.
The attention mechanism learns which feature combinations are important.

## How It Works

The interacting layer is the core component of AutoInt that learns high-order feature
interactions through multi-head self-attention. Each layer captures different orders
of interactions between features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InteractingLayer(Int32,Int32,Nullable<Int32>,Boolean,Double)` | Initializes an interacting layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Forward pass through the interacting layer. |
| `GetAttentionScores` | Gets attention scores for interpretability. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` | Resets internal state. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates parameters using gradient descent. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

