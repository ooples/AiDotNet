---
title: "ParallelBatchLoader<TBatch>"
description: "Provides parallel batch loading with multiple workers for improved throughput."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Provides parallel batch loading with multiple workers for improved throughput.

## For Beginners

Training neural networks often involves:

1. Loading data from disk
2. Preprocessing (augmentation, normalization)
3. GPU training

With single-threaded loading, the GPU waits while data is prepared.
With parallel loading, multiple workers prepare batches simultaneously,
keeping the GPU constantly fed with data.

Example:

## How It Works

ParallelBatchLoader uses multiple worker threads to prepare batches in parallel,
similar to PyTorch's DataLoader with num_workers > 0. This can significantly
improve training throughput when batch preparation is CPU-bound.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ParallelBatchLoader(Func<IEnumerable<Int32>>,Func<Int32[],>,Int32,Nullable<Int32>,Nullable<Int32>)` | Initializes a new instance of the ParallelBatchLoader class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumWorkers` | Gets the number of parallel workers. |
| `PrefetchCount` | Gets the prefetch count. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Disposes the parallel batch loader. |
| `GetBatchesAsync(CancellationToken)` | Iterates through batches using parallel workers. |

