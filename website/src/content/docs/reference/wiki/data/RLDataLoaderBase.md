---
title: "RLDataLoaderBase<T>"
description: "Abstract base class for RL data loaders providing common reinforcement learning functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Loaders`

Abstract base class for RL data loaders providing common reinforcement learning functionality.

## For Beginners

This base class handles common RL operations:

- Stepping through the environment collecting experiences
- Storing experiences in a replay buffer
- Sampling batches for training

Concrete implementations extend this to work with specific environments or
provide specialized experience collection strategies.

## How It Works

RLDataLoaderBase provides shared implementation for all RL data loaders including:

- Environment interaction management
- Replay buffer management
- Episode running and experience collection
- Batch sampling for training

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RLDataLoaderBase(IEnvironment<>,IReplayBuffer<,Vector<>,Vector<>>,Int32,Int32,Int32,Boolean,Nullable<Int32>)` | Initializes a new instance of the RLDataLoaderBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` |  |
| `CurrentEpisode` |  |
| `Environment` |  |
| `Episodes` |  |
| `HasNext` |  |
| `MaxStepsPerEpisode` |  |
| `MinExperiencesBeforeTraining` |  |
| `ReplayBuffer` |  |
| `TotalCount` |  |
| `TotalSteps` |  |
| `Verbose` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddExperience(Experience<,Vector<>,Vector<>>)` |  |
| `CanTrain(Int32)` |  |
| `GetBatches(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>)` |  |
| `GetBatchesAsync(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>,Int32,CancellationToken)` |  |
| `GetNextBatch` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `OnReset` |  |
| `ResetTraining` |  |
| `RunEpisode(IRLAgent<>)` |  |
| `RunEpisodes(Int32,IRLAgent<>)` |  |
| `SampleBatch(Int32)` |  |
| `SelectRandomAction` | Selects a random action for exploration. |
| `SetSeed(Int32)` |  |
| `TryGetNextBatch(Experience<,Vector<>,Vector<>>)` |  |
| `UnloadDataCore` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations helper for type T. |
| `_randomLock` | Lock object for thread-safe access to _random during batch generation. |

