---
title: "ZeRO2Model<T, TInput, TOutput>"
description: "Implements ZeRO Stage 2 model wrapper - shards optimizer states and gradients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements ZeRO Stage 2 model wrapper - shards optimizer states and gradients.

## For Beginners

This implements ZeRO Stage 2, which saves even more memory than ZeRO-1. The model parameters
are still fully replicated (like DDP and ZeRO-1), but now both the optimizer state AND the
gradients are split across processes. After computing gradients, they're immediately reduced
and scattered so each process only keeps its portion.

## How It Works

**Strategy Overview:**
ZeRO Stage 2 builds on ZeRO-1 by additionally sharding gradients across processes.
Parameters are still replicated for the forward pass, but gradients are reduced and scattered
(ReduceScatter) so each process only stores a portion. This saves significant memory compared
to ZeRO-1, especially for large models.

Think of it like a team where everyone has the full playbook (parameters), but when taking
notes during practice (gradients), they divide up the note-taking so each person is responsible
for recording only certain plays. This saves everyone from having to write everything down.

**Use Cases:**

- Larger models where gradient memory becomes significant
- Want substantial memory savings with moderate communication cost
- Preparing for ZeRO-3/FSDP migration

**Trade-offs:**

- Memory: Very Good - saves both optimizer states and gradients
- Communication: Moderate - uses ReduceScatter instead of AllReduce
- Complexity: Moderate - gradient sharding adds some complexity
- Best for: Large models where gradient memory is significant

Example:

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterDeltaShard` | Gets the local parameter delta shard for this rank after synchronization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AllGatherParameterShards` | Reconstructs full parameters by gathering parameter shards from all ranks. |
| `Clone` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `LoadModel(String)` |  |
| `Predict()` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SynchronizeGradients` | Synchronizes gradients using ReduceScatter - each process gets its shard of reduced gradients. |
| `Train(,)` |  |
| `WithParameters(Vector<>)` |  |

