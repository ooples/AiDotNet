---
title: "RetNetLayer<T>"
description: "Implements the RetNet (Retentive Network) layer from Sun et al., 2023."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the RetNet (Retentive Network) layer from Sun et al., 2023.

## For Beginners

RetNet is designed to be a successor to the Transformer architecture.

Think of a Transformer's attention as: "For every word, look at ALL other words and decide what's
important." This is powerful but expensive (quadratic cost).

RetNet replaces this with a "retention" mechanism that works like a fading memory:

- Recent words are remembered clearly (high weight)
- Older words gradually fade away (exponential decay)
- Different heads "forget" at different speeds:
* Some heads have long memory (gamma close to 1.0) - they capture long-range patterns
* Some heads have short memory (gamma around 0.97) - they capture local patterns

The big advantage: RetNet can be computed three ways:

1. Parallel mode (for training): process the whole sequence at once, like a Transformer
2. Recurrent mode (for inference): process one token at a time, like an RNN - O(1) per step
3. Chunkwise mode: a hybrid that balances speed and parallelism

This means RetNet trains like a Transformer but generates text like an RNN, getting the best
of both worlds.

## How It Works

RetNet introduces a multi-scale retention mechanism that replaces softmax attention with an
exponential decay approach, supporting parallel, recurrent, and chunkwise computation modes.
This implementation uses the parallel formulation for training:

where D is a causal decay mask with D_ij = gamma^(i-j) for i >= j and 0 otherwise.

The architecture:

The multi-scale retention is the key innovation: each head operates at a different time scale
via its own decay rate gamma_h. Heads with gamma close to 1.0 retain long-range context, while
heads with smaller gamma focus on local context. This naturally creates a multi-scale representation
without requiring positional encodings.

The decay rates are initialized using the formula from the paper:

which spaces the decay rates logarithmically between approximately 0.97 and 0.9999, ensuring
that heads cover a wide range of temporal scales.

**Reference:** Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models", 2023.
https://arxiv.org/abs/2307.08621

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RetNetLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new RetNet (Retentive Network) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumHeads` | Gets the number of retention heads. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSiLUDerivative(Tensor<>)` | Computes the derivative of the SiLU (Swish) activation function. |
| `Forward(Tensor<>)` |  |
| `GetDecayRates` | Gets the per-head decay rates (gammas) for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GroupNormBackward(Tensor<>,Tensor<>,Int32,Int32)` | Backward pass through group normalization. |
| `GroupNormForward(Tensor<>,Int32,Int32)` | Applies group normalization (per-head LayerNorm) to the retention output. |
| `InitializeTensor2D(Tensor<>)` | Xavier/Glorot initialization for 2D weight tensors. |
| `MultiScaleRetentionForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Computes multi-scale retention in parallel mode: Retention_h(X) = (Q_h K_h^T odot D_h) V_h where D_h[i,j] = gamma_h^(i-j) for i >= j, 0 otherwise. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

