---
title: "FSDPModel<T, TInput, TOutput>"
description: "Implements FSDP (Fully Sharded Data Parallel) model wrapper that shards parameters across multiple processes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements FSDP (Fully Sharded Data Parallel) model wrapper that shards parameters across multiple processes.

## For Beginners

This class implements FSDP (Fully Sharded Data Parallel), which makes any model work across multiple GPUs or machines
with maximum memory efficiency. It automatically handles:

- Splitting ALL model components (parameters, gradients, optimizer states) across processes
- Gathering parameters only when needed for forward/backward pass
- Releasing parameters immediately after use to save memory
- Averaging gradients across all processes during training

## How It Works

**Strategy Overview:**
FSDP (Fully Sharded Data Parallel) is PyTorch's implementation of the ZeRO-3 optimization strategy.
It shards model parameters, gradients, and optimizer states across all processes, achieving maximum
memory efficiency. Parameters are gathered just-in-time for forward/backward passes and then released.

Think of it like a team project where each person holds part of the solution, but unlike DDP,
FSDP only shares the full model temporarily when absolutely needed, then immediately goes back
to holding just their piece. This saves a lot of memory!

**Use Cases:**

- Training very large models that don't fit in a single GPU's memory
- Maximizing memory efficiency for multi-GPU training
- Scaling to hundreds or thousands of GPUs

**Trade-offs:**

- Memory: Excellent - shards everything (parameters + gradients + optimizer states)
- Communication: Higher - requires AllGather for each forward/backward pass
- Complexity: Moderate - automatic just-in-time parameter gathering
- Best for: Very large models, memory-constrained scenarios

Example:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FSDPModel(IFullModel<,,>,IShardingConfiguration<>)` | Creates a new FSDP model wrapping an existing model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `GetActiveFeatureIndices` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `IsFeatureUsed(Int32)` |  |
| `LoadModel(String)` |  |
| `Predict()` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` |  |
| `SynchronizeGradients` |  |
| `Train(,)` |  |
| `WithParameters(Vector<>)` |  |

