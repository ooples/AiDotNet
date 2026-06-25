---
title: "ABCLayer<T>"
description: "Implements the ABC (Attention with Bounded-memory Control) layer from Peng et al., 2022."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the ABC (Attention with Bounded-memory Control) layer from Peng et al., 2022.

## For Beginners

ABC is like having a fixed number of filing cabinet drawers (slots)
for storing information as you read through a long document.

Imagine you have 32 drawers and you're reading a book:

- At each word, you decide which drawers are most relevant (via attention scores)
- You file information about the word into those drawers (competitive writing)
- Old information gradually fades from drawers (forget gate)
- When you need to answer a question, you look through the drawers (reading)

The "competitive" part is crucial: if many words want to use the same drawer,
softmax ensures the most relevant one gets priority. This is what "bounded-memory
control" means -- you never need more drawers than the fixed number, no matter
how long the book is.

Compare this to:

- Standard attention: You keep all words accessible (expensive for long books)
- Linear attention: You maintain a summary matrix (unbounded growth in rank)
- ABC: You maintain exactly numSlots drawers of information (bounded)

## How It Works

ABC uses a fixed-size set of memory "slots" with a competitive attention mechanism. Input tokens
compete for writing to slots via softmax attention scores, and a forget mechanism clears stale
slot content. This bounds memory usage regardless of sequence length while maintaining the ability
to selectively store and retrieve information.

The architecture:

The key insight is competitive slot access: tokens compete to write into a bounded number of
memory slots via softmax. This naturally implements a form of memory management where the most
relevant information gets stored and stale information is gradually forgotten. Unlike unbounded
linear attention states, the fixed slot count guarantees constant memory.

**Reference:** Peng et al., "ABC: Attention with Bounded-memory Control", 2022.
https://arxiv.org/abs/2110.02488

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ABCLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new ABC (Attention with Bounded-memory Control) layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `NumSlots` | Gets the number of memory slots per head. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetSlotKeys` | Gets the slot key embeddings for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SlotCompetitionForward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Competitive slot write and read mechanism. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

