---
title: "IBatchIterable<TBatch>"
description: "Defines capability to iterate through data in batches."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines capability to iterate through data in batches.

## For Beginners

Instead of feeding your model one example at a time,
batching groups multiple examples together. Training in batches is faster
(more efficient GPU usage) and often leads to better learning (smoother gradients).

## How It Works

Data loaders that implement this interface can provide data in batches,
which is the standard way to process data during model training.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the number of samples per batch. |
| `HasNext` | Gets whether there are more batches available in the current iteration. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBatches(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>)` | Iterates through all batches in the dataset using lazy evaluation. |
| `GetBatchesAsync(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>,Int32,CancellationToken)` | Asynchronously iterates through all batches with prefetching support. |
| `GetNextBatch` | Gets the next batch of data. |
| `TryGetNextBatch()` | Attempts to get the next batch without throwing if unavailable. |

