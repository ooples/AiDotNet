---
title: "DDPModel<T, TInput, TOutput>"
description: "Implements DDP (Distributed Data Parallel) model wrapper for distributed training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements DDP (Distributed Data Parallel) model wrapper for distributed training.

## For Beginners

This class implements DDP (Distributed Data Parallel), the simplest and most popular way to train
models across multiple GPUs or machines. Unlike FSDP which shards parameters, DDP keeps a complete
copy of the model on each process. It automatically handles:

- Keeping full model parameters on each process (no sharding)
- Averaging gradients across all processes after backward pass
- Ensuring all model replicas stay synchronized

## How It Works

**Strategy Overview:**
DDP (Distributed Data Parallel) is the most common and straightforward distributed training strategy.
Each process maintains a full replica of the model. During training, gradients are synchronized
across all processes using AllReduce, ensuring all replicas stay identical. This is PyTorch's
default distributed training strategy.

Think of it like multiple chefs each making the full recipe. After each step, they compare notes
and average their learnings, so everyone stays on the same page. This is simpler than FSDP where
each person only knows part of the recipe.

**Use Cases:**

- Standard multi-GPU training where model fits in single GPU memory
- When communication is fast (NVLink, InfiniBand)
- Simpler debugging than FSDP (full model on each process)
- Default choice for most distributed training scenarios

**Trade-offs:**

- Memory: Moderate - each process stores full model (parameters replicated)
- Communication: Low - only gradients synchronized (AllReduce after backward)
- Complexity: Low - simplest distributed strategy
- Best for: Models that fit in single GPU memory, fast interconnects

Example:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DDPModel(IFullModel<,,>,IShardingConfiguration<>)` | Creates a new DDP model wrapping an existing model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `InitializeSharding` | Initializes DDP - no actual parameter sharding, each process keeps full parameters. |
| `LoadModel(String)` |  |
| `Predict()` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SynchronizeGradients` | Synchronizes gradients across all processes using AllReduce. |
| `Train(,)` |  |
| `WithParameters(Vector<>)` |  |

