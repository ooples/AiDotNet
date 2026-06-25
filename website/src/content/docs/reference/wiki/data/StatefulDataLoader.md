---
title: "StatefulDataLoader<T, TInput, TOutput>"
description: "Wraps any `InMemoryDataLoader` with checkpoint/resume support."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Wraps any `InMemoryDataLoader` with checkpoint/resume support.

## For Beginners

Wrap your data loader with this class to enable
saving and restoring the exact position during training:

## How It Works

Inspired by PyTorch's StatefulDataLoader (torchdata 2025), this wrapper adds
mid-epoch checkpointing to any in-memory data loader. The state can be serialized
and restored for fault-tolerant training on large datasets.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StatefulDataLoader(InputOutputDataLoaderBase<,,>)` | Creates a stateful wrapper around an existing data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Epoch` | Gets the current epoch number. |
| `FeatureCount` |  |
| `Features` |  |
| `HasNext` |  |
| `IsShuffled` |  |
| `Labels` |  |
| `Name` |  |
| `OutputDimension` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBatches(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>)` |  |
| `GetBatchesAsync(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>,Int32,CancellationToken)` |  |
| `GetNextBatch` |  |
| `GetState` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `LoadState(DataLoaderCheckpoint)` |  |
| `OnEpochStart` | Called when starting a new epoch. |
| `Shuffle(Nullable<Int32>)` |  |
| `Split(Double,Double,Nullable<Int32>)` |  |
| `TryGetNextBatch(ValueTuple<,>)` |  |
| `UnloadDataCore` |  |
| `Unshuffle` |  |

