---
title: "IShardedModel<T, TInput, TOutput>"
description: "Defines the contract for models that support distributed training with parameter sharding."
section: "API Reference"
---

`Interfaces` · `AiDotNet.DistributedTraining`

Defines the contract for models that support distributed training with parameter sharding.

## For Beginners

A sharded model is like having a team working on a large puzzle together.
Instead of one person holding all the puzzle pieces (parameters), each person
holds only a portion. When someone needs to see the full picture, everyone
shares their pieces (AllGather). When the team learns something new, everyone
combines their learnings (AllReduce).

## How It Works

This allows training models that are too large to fit on a single GPU or machine.

## Properties

| Property | Summary |
|:-----|:--------|
| `LocalParameterShard` | Gets the portion of parameters owned by this process. |
| `Rank` | Gets the rank of this process in the distributed group. |
| `ShardingConfiguration` | Gets the configuration for this sharded model. |
| `WorldSize` | Gets the total number of processes in the distributed group. |
| `WrappedModel` | Gets the underlying wrapped model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GatherFullParameters` | Gets the full set of parameters by gathering from all processes. |
| `SynchronizeGradients` | Synchronizes gradients across all processes using AllReduce. |

