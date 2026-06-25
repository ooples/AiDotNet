---
title: "FSDPOptimizer<T, TInput, TOutput>"
description: "Implements FSDP (Fully Sharded Data Parallel) optimizer wrapper that coordinates optimization across multiple processes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements FSDP (Fully Sharded Data Parallel) optimizer wrapper that coordinates optimization across multiple processes.

## For Beginners

This class wraps any existing optimizer (like Adam, SGD, etc.) and makes it work with FSDP strategy
across multiple GPUs or machines. It automatically handles:

- Synchronizing gradients across all processes
- Sharding optimizer states (momentum, variance) to save memory
- Coordinating parameter updates
- Ensuring all processes stay in sync

## How It Works

**Strategy Overview:**
FSDP optimizer works in conjunction with FSDPModel to provide full sharding of optimizer states.
This means momentum buffers, variance estimates, and all other optimizer-specific state are sharded
across processes, minimizing memory usage while maintaining training effectiveness.

Think of it like a team of coaches working together - each coach has their own expertise
(the wrapped optimizer), but they share only the essential information and keep their detailed
notes (optimizer states) private to save space.

**Use Cases:**

- Training very large models with optimizers that have significant state (Adam, RMSprop)
- Maximizing memory efficiency when using stateful optimizers
- Scaling to hundreds or thousands of GPUs

**Trade-offs:**

- Memory: Excellent - shards optimizer states across processes
- Communication: Moderate - syncs gradients and occasional state synchronization
- Complexity: Moderate - automatic state sharding
- Best for: Large models with stateful optimizers (Adam, RMSprop, etc.)

Example:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FSDPOptimizer(IOptimizer<,,>,IShardingConfiguration<>)` | Creates a new FSDP optimizer wrapping an existing optimizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` |  |
| `GetOptions` |  |
| `LoadModel(String)` |  |
| `Optimize(OptimizationInputData<,,>)` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `ShouldEarlyStop` |  |
| `SynchronizeOptimizerState` |  |

