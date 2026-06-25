---
title: "ShardedModelBase<T, TInput, TOutput>"
description: "Provides base implementation for distributed models with parameter sharding."
section: "API Reference"
---

`Base Classes` · `AiDotNet.DistributedTraining`

Provides base implementation for distributed models with parameter sharding.

## For Beginners

This is the foundation that all distributed models build upon.

Think of this as a template for splitting a big model across multiple computers or GPUs.
It handles common tasks like:

- Dividing model parameters into chunks (sharding)
- Collecting all chunks when needed (gathering)
- Sharing learning updates across all processes (gradient sync)
- Saving and loading distributed models

Specific types of distributed models (like fully sharded or hybrid sharded) inherit
from this and add their own strategies. This prevents code duplication and ensures
all distributed models work consistently.

## How It Works

This abstract class implements common functionality for all sharded models,
including parameter management, sharding logic, gradient synchronization, and
integration with the model serialization system. Derived classes can customize
the sharding strategy, communication pattern, or add optimization-specific features.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShardedModelBase(IFullModel<,,>,IShardingConfiguration<>)` | Initializes a new instance of the ShardedModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `LocalParameterShard` |  |
| `ParameterCount` |  |
| `Rank` |  |
| `ShardingConfiguration` |  |
| `SupportsParameterInitialization` |  |
| `WorldSize` |  |
| `WrappedModel` |  |
| `WrappedModelInternal` | Protected access to wrapped model for derived classes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `Clone` |  |
| `ComputeGradients(,,ILossFunction<>)` |  |
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `Dispose` |  |
| `Dispose(Boolean)` | Disposes the wrapped sharded model. |
| `EnsureShardingInitialized` | Ensures that sharding has been initialized. |
| `GatherFullParameters` |  |
| `GetActiveFeatureIndices` |  |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` |  |
| `GetInputShape` |  |
| `GetModelMetadata` |  |
| `GetOutputShape` |  |
| `GetParameters` |  |
| `InitializeSharding` | Initializes parameter sharding by dividing parameters across processes. |
| `InvalidateCache` | Invalidates the cached full parameters, forcing a re-gather on next access. |
| `IsFeatureUsed(Int32)` |  |
| `LoadModel(String)` |  |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `OnBeforeInitializeSharding` | Called before InitializeSharding to allow derived classes to set up state. |
| `Predict()` |  |
| `SanitizeParameters(Vector<>)` |  |
| `SaveModel(String)` |  |
| `SaveState(Stream)` | Saves the model's current state to a stream. |
| `Serialize` |  |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |
| `SynchronizeGradients` |  |
| `Train(,)` |  |
| `UpdateLocalShardFromFull(Vector<>)` | Updates the local parameter shard from the full parameter vector. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CachedFullParameters` | Cached full parameters to avoid repeated gathering. |
| `Config` | The sharding configuration containing communication backend and settings. |
| `LocalShard` | The local parameter shard owned by this process. |
| `NumOps` | Provides numeric operations for type T. |
| `ShardSize` | Size of this process's parameter shard. |
| `ShardStartIndex` | Starting index of this process's shard in the full parameter vector. |
| `_isShardingInitialized` | Flag indicating whether sharding has been initialized. |
| `_wrappedModel` | The wrapped model that this sharded model delegates to. |

