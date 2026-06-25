---
title: "ShardedOptimizerBase<T, TInput, TOutput>"
description: "Provides base implementation for distributed optimizers with parameter sharding."
section: "API Reference"
---

`Base Classes` · `AiDotNet.DistributedTraining`

Provides base implementation for distributed optimizers with parameter sharding.

## For Beginners

This is the foundation that all distributed optimizers build upon.

Think of this as a template for coordinating optimization across multiple computers or GPUs.
It handles common tasks like:

- Wrapping regular optimizers to work in distributed mode
- Syncing parameters across all processes after updates
- Making sure all processes agree on when to stop training
- Saving and loading distributed optimizer state

Specific types of distributed optimizers (like data-parallel or ZeRO) inherit from
this and add their own strategies. This prevents code duplication and ensures all
distributed optimizers work consistently.

## How It Works

This abstract class implements common functionality for all sharded optimizers,
including optimizer wrapping, parameter synchronization, consensus-based early stopping,
and serialization. Derived classes can customize the optimization strategy, implement
different sharding approaches (FSDP, ZeRO, etc.), or add optimizer-specific features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShardedOptimizerBase(IOptimizer<,,>,IShardingConfiguration<>)` | Initializes a new instance of the ShardedOptimizerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LastComputedGradients` | Gets the gradients computed during the last optimization step. |
| `Rank` |  |
| `ShardingConfiguration` |  |
| `WorldSize` |  |
| `WrappedOptimizer` |  |
| `WrappedOptimizerInternal` | Protected access to wrapped optimizer for derived classes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,IFullModel<,,>)` | Applies pre-computed gradients to a model's parameters. |
| `Deserialize(Byte[])` |  |
| `GetDynamicShapeInfo` |  |
| `GetInputShape` |  |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `LoadModel(String)` |  |
| `Optimize(OptimizationInputData<,,>)` |  |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SetModel(IFullModel<,,>)` |  |
| `ShouldEarlyStop` |  |
| `SynchronizeOptimizerState` |  |
| `SynchronizeParameters(IFullModel<,,>)` | Synchronizes model parameters across all processes using AllReduce with averaging. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Config` | The sharding configuration containing communication backend and settings. |
| `NumOps` | Provides numeric operations for type T. |
| `_wrappedOptimizer` | The wrapped optimizer that this sharded optimizer delegates to. |

