---
title: "KimiLinearAttentionLayer<T>"
description: "Implements the Kimi KDA (Key-Value Driven Gated Linear Attention) layer from the \"Kimi-VL Technical Report\" (Kimi Team, 2025, arXiv:2510.26692)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Kimi KDA (Key-Value Driven Gated Linear Attention) layer from the
"Kimi-VL Technical Report" (Kimi Team, 2025, arXiv:2510.26692).

## For Beginners

Kimi KDA is a way to read through a sequence and build up a
memory of what you've seen, with a smart forgetting mechanism.

Think of it like taking notes while reading a book:

- At each word, you have a "key" (what topic this is about) and a "value" (what it says)
- The gate is like asking: "Does this new information FIT with what I already know?"
- If key and value are very aligned (k^T * v is large): "This is strong, clear info" -> gate near 1
- If they're not aligned: "This is ambiguous or contradictory" -> gate lower
- When the gate is high: keep most of old notes (g * S) and add new info (k * v^T)
- When the gate is low: forget more old notes, making room for new patterns

This is different from other approaches that gate based on the INPUT position:

- Position-based gate: "I'm at word 50, so I should forget some old stuff" (doesn't know WHAT to forget)
- KV-driven gate: "This new key-value pair conflicts with stored patterns, so selectively forget" (content-aware)

The result is more intelligent memory management that adapts to the actual information content.

## How It Works

Kimi KDA is a gated linear attention mechanism where the gate is computed from the interaction
between keys and values rather than from a separate learned projection. The gate signal
g_t = sigma(k_t^T * v_t + bias) captures whether the current key-value pair contains
information that conflicts with or reinforces the current state. This KV-driven gating
replaces the typical input-driven gating found in other linear attention variants.

The architecture:

The key insight is that the gate should reflect the CONTENT of what's being stored, not just
where in the sequence we are. When k_t and v_t are highly aligned (high dot product), the
information is coherent and can be safely accumulated. When they conflict with stored patterns,
the gate should allow partial forgetting. This KV-driven approach automatically detects when
new information should overwrite old information.

**Reference:** Kimi Team, "Kimi-VL Technical Report", 2025.
https://arxiv.org/abs/2510.26692

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KimiLinearAttentionLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Kimi KDA (Key-Value Driven Gated Linear Attention) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of heads. |
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `KVGatedRecurrence(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | KV-driven gated linear attention recurrence. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SigmoidScalar()` | Computes sigmoid for a scalar value: 1 / (1 + exp(-x)). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

