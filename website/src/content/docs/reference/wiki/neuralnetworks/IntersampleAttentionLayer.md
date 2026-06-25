---
title: "IntersampleAttentionLayer<T>"
description: "Intersample (Row) Attention for SAINT architecture."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Intersample (Row) Attention for SAINT architecture.

## For Beginners

While column attention looks at relationships between features
within a single sample, intersample attention looks at relationships between different
samples in the batch:

- Column attention: "How does age relate to income for this person?"
- Intersample attention: "How does this person compare to others in the batch?"

This allows the model to learn from the distribution of data, not just individual samples.

## How It Works

Intersample attention allows samples in a batch to attend to each other,
enabling the model to learn relationships between different data points.
This is a key innovation in SAINT that helps with semi-supervised learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IntersampleAttentionLayer(Int32,Int32,Double)` | Initializes intersample attention. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsureInitialized` | Auto-generated EnsureInitialized: registers sub-layers (cheap), then delegates to base for weight allocation. |
| `EnsureSubLayersRegistered` | Registers discovered sub-layer fields exactly once. |
| `Forward(Tensor<>)` | Forward pass through intersample attention. |
| `GetMetadata` | Persists the head count and dropout rate so the Clone serialize→deserialize round-trip can reconstruct the layer (embedding dim is recoverable from the saved shape). |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` | Resets internal state. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates parameters. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

