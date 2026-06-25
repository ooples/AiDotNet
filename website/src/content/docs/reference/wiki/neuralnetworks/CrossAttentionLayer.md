---
title: "CrossAttentionLayer<T>"
description: "Implements cross-attention for conditioning diffusion models on text or other context."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements cross-attention for conditioning diffusion models on text or other context.

## For Beginners

Cross-attention is how the model "reads" the text prompt.

- Queries: "What should I generate at each position?"
- Keys/Values: "What does the text describe?"
- Output: Spatial features modified to match the text description

## How It Works

Cross-attention differs from self-attention in that queries come from spatial features
while keys and values come from conditioning (text embeddings). This is the core mechanism
that allows text-to-image models like Stable Diffusion to follow prompts.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CrossAttentionLayer(Int32,Int32,Int32,Int32)` | Creates a new cross-attention layer for conditioning. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsGpuExecution` | Gets a value indicating whether this layer can execute on GPU. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>)` | Forward pass for self-attention (not typically used for cross-attention). |
| `Forward(Tensor<>[])` | Forward pass with multiple inputs for cross-attention. |
| `ForwardCrossAttention(Tensor<>,Tensor<>)` | Forward pass with separate query and context tensors. |
| `ForwardGpu(Tensor<>[])` | GPU-resident forward pass for cross-attention with multiple inputs. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputPortsCache` | Declares named input ports: "query" (required) and "context" (optional, defaults to query for self-attention). |
| `_originalQueryShape` | Stores the original query shape for any-rank tensor support. |

