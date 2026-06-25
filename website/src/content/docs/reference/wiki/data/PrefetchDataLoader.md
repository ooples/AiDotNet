---
title: "PrefetchDataLoader<TBatch>"
description: "Wraps a batch-producing function with asynchronous prefetching for pipelined data loading."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Wraps a batch-producing function with asynchronous prefetching for pipelined data loading.

## How It Works

Prefetching overlaps data loading with model computation by preparing the next N batches
in advance on a background thread. This hides I/O latency and keeps the GPU busy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrefetchDataLoader(PrefetchDataLoaderOptions)` | Creates a new prefetch data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BufferedCount` | Gets the number of batches currently buffered. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `Prefetch(IEnumerable<>)` | Starts prefetching batches from the provided source. |
| `Stop` | Stops prefetching and cleans up resources. |

