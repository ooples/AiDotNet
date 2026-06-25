---
title: "DeltaFormerLayer<T>"
description: "Implements the DeltaFormer layer from \"An Associative Memory Perspective on Transformers and DeltaNet\" (Li and Papailiopoulos, 2025, arXiv:2505.19488)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the DeltaFormer layer from "An Associative Memory Perspective on Transformers and DeltaNet"
(Li and Papailiopoulos, 2025, arXiv:2505.19488).

## For Beginners

DeltaFormer combines two different ways of processing information,
alternating between them like two specialized workers on an assembly line.

Imagine studying for an exam:

- The "delta rule" worker is the note-taker: they read through material and update their notes,

only writing down what's NEW or DIFFERENT from what they already have. This is the "delta" —
the correction needed to update existing knowledge.

- The "attention" worker is the test-taker: when asked a question, they search through all

available information to find the best answer.

By alternating these two operations:

1. Delta rule layers consolidate information into compact, reusable memories
2. Attention layers retrieve from those consolidated memories efficiently

This is more effective than using either approach alone. Pure attention has no persistent memory
between queries; pure delta rule has less flexible retrieval.

## How It Works

DeltaFormer views transformers through an associative memory lens, proposing a hybrid architecture
that alternates between standard softmax attention layers and delta rule layers. The attention layers
handle retrieval of stored associations, while the delta rule layers handle memory consolidation by
writing only the correction needed to update the fast weight matrix.

The architecture:

The key insight is that attention and delta rule are complementary: attention is excellent at
one-shot retrieval (given a query, find the best match in context), while the delta rule excels at
memory consolidation (incrementally building a reusable association table). Alternating them gets
the best of both worlds: consolidated memories that are efficiently retrievable.

**Reference:** Li and Papailiopoulos, "An Associative Memory Perspective on Transformers and DeltaNet", 2025.
https://arxiv.org/abs/2505.19488

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeltaFormerLayer(Int32,Int32,Int32,Boolean,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new DeltaFormer layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` |  |
| `SupportsTraining` |  |
| `UseDeltaRule` | Gets whether this layer uses the delta rule (true) or standard attention (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeltaRuleForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Delta rule forward: S_t = S_{t-1} + (v_t - S_{t-1} * k_t) * k_t^T. |
| `Forward(Tensor<>)` |  |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SoftmaxAttentionForward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Standard softmax attention: softmax(Q*K^T / sqrt(d)) * V. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

