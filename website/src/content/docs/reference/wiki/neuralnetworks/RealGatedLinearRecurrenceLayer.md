---
title: "RealGatedLinearRecurrenceLayer<T>"
description: "Implements the Real-Gated Linear Recurrence Unit (RG-LRU) from Google DeepMind's Griffin architecture."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Real-Gated Linear Recurrence Unit (RG-LRU) from Google DeepMind's Griffin architecture.

## For Beginners

The RG-LRU is like a learnable "leaky bucket" for information.

Imagine each position in your hidden state as a bucket:

- The recurrence gate (r) controls how much water leaks out each step (memory decay)
- The input gate (i) controls how much new water pours in
- The sqrt(1 - a^2) factor ensures the bucket never overflows or runs dry

This is simpler than Mamba (no Conv1D, no SSM parameters B/C) but surprisingly effective.
Google's RecurrentGemma models (2B, 9B) use this architecture and achieve competitive
performance with Transformer-based Gemma models.

## How It Works

The RG-LRU is a gated linear recurrence that serves as the core sequence mixing mechanism in
the Griffin and Hawk architectures. It uses input-dependent gating to control both the recurrence
decay and the input contribution, providing selective memory similar to Mamba but through a
different mathematical formulation.

The recurrence is:

The sqrt(1 - a_t^2) factor ensures the recurrence preserves signal magnitude, preventing
vanishing or exploding states.

Griffin combines RG-LRU with local attention in a hybrid architecture. This layer implements
the RG-LRU component which can be used standalone or as part of a hybrid.

**Reference:** De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", 2024.
https://arxiv.org/abs/2402.19427

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealGatedLinearRecurrenceLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Real-Gated Linear Recurrence Unit (RG-LRU) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (input/output width). |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `RecurrenceDimension` | Gets the recurrence dimension (hidden state width). |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GatedRecurrenceForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Implements the gated linear recurrence with magnitude-preserving update. |
| `GetDecayParameter` | Gets the decay parameter for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

