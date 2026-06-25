---
title: "RWKVLayer<T>"
description: "Implements the RWKV (Receptance Weighted Key Value) layer, a linear attention RNN from Peng et al., 2024."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the RWKV (Receptance Weighted Key Value) layer, a linear attention RNN from Peng et al., 2024.

## For Beginners

RWKV is like a clever hybrid between a Transformer and an RNN.

Imagine you're summarizing a conversation:

- A Transformer re-reads the entire conversation for every new sentence (expensive but thorough)
- An RNN keeps a running summary and just adds new info (cheap but may forget)
- RWKV keeps a running summary (like an RNN) but uses a smart weighting scheme

so recent information is weighted more heavily, like how you naturally pay more attention
to what was just said

The "token shift" mechanism is like looking at both the current word and the previous word
to decide what's important - a simple but effective trick.

Used by RWKV Foundation models (Eagle v5, Finch v6, Goose v7) which achieve competitive
performance with Transformers at much lower inference cost.

## How It Works

RWKV combines the training parallelism of Transformers with the efficient inference of RNNs.
It uses a linear attention mechanism with data-dependent decay, avoiding the quadratic complexity
of standard attention while maintaining competitive quality for language modeling.

The architecture consists of two mixing modules per layer:

RWKV v6 (Finch) adds data-dependent linear interpolation for the token-shift mixing coefficients,
making mu_r, mu_k, mu_v input-dependent rather than fixed learned parameters.

**Reference:** Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence", 2024.
https://arxiv.org/abs/2404.05892

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RWKVLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new RWKV layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` | Training IS supported. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyLayerNorm(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Applies layer normalization. |
| `ChannelMixingForward(Tensor<>,Int32,Int32)` | Channel mixing forward: squared ReLU with receptance gating. |
| `Forward(Tensor<>)` |  |
| `GetOutputWeights` | Gets a copy of the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetReceptanceWeights` | Gets a copy of the receptance projection weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `TimeMixingForward(Tensor<>,Int32,Int32)` | Time mixing forward: token shift + linear attention with exponential decay. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

