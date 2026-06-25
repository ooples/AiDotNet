---
title: "StreamingDataLoaderBase<T, TInput, TOutput>"
description: "Abstract base class for streaming data loaders that process data on-demand."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Loaders`

Abstract base class for streaming data loaders that process data on-demand.

## For Beginners

When working with huge datasets (millions of images,
terabytes of text), you can't load everything into memory at once. This base class
handles the complexity of streaming data efficiently while you focus on implementing
the actual data reading logic.

## How It Works

StreamingDataLoaderBase provides the foundation for data loaders that read data on-demand
rather than loading everything into memory. This is essential for:

- Large datasets that don't fit in RAM
- Real-time data streams
- Memory-efficient training pipelines

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StreamingDataLoaderBase(Int32,Int32,Int32)` | Initializes a new instance of the StreamingDataLoaderBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumWorkers` |  |
| `PrefetchCount` |  |
| `SampleCount` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateSamples(IList<ValueTuple<,>>)` | Aggregates multiple samples into a batch. |
| `GetBatches(Boolean,Boolean,Nullable<Int32>)` |  |
| `GetBatchesAsync(Boolean,Boolean,Nullable<Int32>,CancellationToken)` |  |
| `GetShuffledIndices(Boolean,Nullable<Int32>)` | Gets indices for iteration, optionally shuffled. |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `ReadSampleAsync(Int32,CancellationToken)` | Reads a single sample by index. |
| `UnloadDataCore` |  |

