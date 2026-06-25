---
title: "IStreamingDataLoader<T, TInput, TOutput>"
description: "Interface for streaming data loaders that process data on-demand without loading all data into memory."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for streaming data loaders that process data on-demand without loading all data into memory.

## For Beginners

When your dataset is too large to fit in RAM (like millions
of images or text documents), you can't load it all at once. Streaming data loaders
solve this by reading data piece by piece as needed during training.

Example usage:

## How It Works

IStreamingDataLoader is designed for datasets that are too large to fit in memory.
Unlike IInputOutputDataLoader which provides Features and Labels properties for all data,
streaming loaders read data on-demand and yield batches through iteration.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for iteration. |
| `NumWorkers` | Gets the number of parallel workers for sample loading. |
| `PrefetchCount` | Gets the number of batches to prefetch for improved throughput. |
| `SampleCount` | Gets the total number of samples in the dataset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBatches(Boolean,Boolean,Nullable<Int32>)` | Iterates through the dataset in batches synchronously. |
| `GetBatchesAsync(Boolean,Boolean,Nullable<Int32>,CancellationToken)` | Iterates through the dataset in batches asynchronously with prefetching. |

