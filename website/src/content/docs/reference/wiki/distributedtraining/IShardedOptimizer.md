---
title: "IShardedOptimizer<T, TInput, TOutput>"
description: "Defines the contract for optimizers that support distributed training with parameter sharding."
section: "API Reference"
---

`Interfaces` · `AiDotNet.DistributedTraining`

Defines the contract for optimizers that support distributed training with parameter sharding.

## For Beginners

A sharded optimizer is like having a team of coaches working together.
Each coach (process) is responsible for updating a portion of the player's (model's) skills.
After each round of practice, the coaches share and combine their improvements to ensure
everyone stays in sync.

## How It Works

This allows optimizing very large models that don't fit on a single GPU.

## Properties

| Property | Summary |
|:-----|:--------|
| `Rank` | Gets the rank of this process in the distributed group. |
| `ShardingConfiguration` | Gets the sharding configuration for this optimizer. |
| `WorldSize` | Gets the total number of processes in the distributed group. |
| `WrappedOptimizer` | Gets the underlying wrapped optimizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SynchronizeOptimizerState` | Synchronizes optimizer state (like momentum buffers) across all processes. |

