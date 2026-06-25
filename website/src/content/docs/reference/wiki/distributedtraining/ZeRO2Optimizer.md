---
title: "ZeRO2Optimizer<T, TInput, TOutput>"
description: "Implements ZeRO Stage 2 optimizer - shards gradients and optimizer states across ranks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements ZeRO Stage 2 optimizer - shards gradients and optimizer states across ranks.

## For Beginners

ZeRO-2 divides the work of storing and updating parameters across processes. Think of it
like a team where each person is responsible for maintaining a specific section of a large
document. Everyone reads the full document (forward pass), but each person only stores and
updates their assigned section (backward pass). Before the next iteration, they share their
sections to reconstruct the full document.

## How It Works

**Strategy Overview:**
True ZeRO-2 implementation using ReduceScatter for gradient sharding. Each rank:

1. Computes local gradients on full parameter set
2. ReduceScatter: reduces gradients AND scatters them (each rank gets a shard)
3. Updates only its shard of parameters using its shard of gradients
4. AllGather: reconstructs full parameters from shards for next forward pass

This saves memory by distributing gradient storage and parameter updates across ranks.

**Use Cases:**

- Large models where gradient memory is significant (billions of parameters)
- Want memory savings beyond DDP
- Good network for AllGather operations
- Works with ANY gradient-based optimizer (SGD, Adam, RMSprop, etc.)

**Trade-offs:**

- Memory: Very Good - gradients and optimizer states sharded (1/N of DDP)
- Communication: ReduceScatter + AllGather (vs AllReduce for DDP)
- Synchronization: Perfect - all ranks reconstruct identical parameters
- Complexity: Moderate - requires parameter sharding logic
- Best for: Large models with limited GPU memory

**Memory Savings vs DDP:**

- DDP: Each rank stores full gradients + full optimizer state
- ZeRO-2: Each rank stores 1/N gradients + 1/N optimizer state
- Savings increase linearly with world size

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ZeRO2Optimizer(IOptimizer<,,>,IShardingConfiguration<>)` | Creates a ZeRO-2 optimizer that shards gradients and optimizer states. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` |  |
| `Optimize(OptimizationInputData<,,>)` |  |
| `Serialize` |  |
| `SynchronizeOptimizerState` |  |

