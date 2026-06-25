---
title: "GatedSlotAttentionLayer<T>"
description: "Implements the Gated Slot Attention (GSA) layer from Li et al., 2024."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements the Gated Slot Attention (GSA) layer from Li et al., 2024.

## For Beginners

GSA is a memory-efficient attention alternative.

Imagine you have a whiteboard with a fixed number of "slots" (rows) for taking notes:

- Standard attention: You compare every word with every other word (expensive for long texts)
- GSA: You maintain a fixed-size "summary board" and update it as you read each word

At each step:

- The "forget gate" decides which old notes to erase (like erasing parts of the whiteboard)
- The "input gate" decides how strongly to write new notes
- The "key" determines WHERE on the board to write
- The "value" determines WHAT to write
- The "query" determines what to READ from the board

Because the board has a fixed number of slots, the memory cost stays constant
regardless of how long the text is, making GSA efficient for very long sequences.

The gating mechanism is crucial: without it, old information would pile up and the
slots would become increasingly noisy. The gates let the model learn to selectively
retain important information and overwrite stale content.

## How It Works

Gated Slot Attention maintains a fixed-size set of "slots" that act as compressed memory,
combining ideas from slot attention (object-centric learning) with gated linear recurrences
for efficient linear-time sequence modeling.

The architecture:

The key difference from standard linear attention: the fixed slot count (n_slots) bounds
memory usage regardless of sequence length, and the dual gating mechanism (forget + input)
provides fine-grained control over information flow. This is analogous to how LSTM gates
control cell state, but applied to a matrix-valued memory (the slots).

**Reference:** Li et al., "Gated Slot Attention for Efficient Linear-Time Sequence Modeling", 2024.
https://arxiv.org/abs/2409.07146

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GatedSlotAttentionLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new Gated Slot Attention layer. |

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
| `GetInitialSlots` | Gets the initial slot embeddings for external inspection. |
| `GetOutputProjectionWeights` | Gets the output projection weights for external inspection. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetQueryWeights` | Gets the query weights for external inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SlotRecurrenceForward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Slot recurrence forward: gated write to slots and query-based read from slots. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

