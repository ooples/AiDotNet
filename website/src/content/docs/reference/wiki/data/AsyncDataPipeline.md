---
title: "AsyncDataPipeline<T>"
description: "Provides async data pipeline operations with prefetching support."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Pipeline`

Provides async data pipeline operations with prefetching support.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AsyncDataPipeline(IAsyncEnumerableProvider<>)` | Initializes a new instance of the AsyncDataPipeline class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Batch(Int32,Boolean)` | Groups elements into batches. |
| `Filter(Func<,Boolean>)` | Filters elements based on a predicate. |
| `GetAsyncEnumerator(CancellationToken)` |  |
| `Map(Func<,>)` | Applies a transformation function to each element. |
| `MapAsync(Func<,CancellationToken,Task<>>)` | Applies an async transformation function to each element. |
| `Prefetch(Int32)` | Prefetches elements in the background for improved performance. |
| `Take(Int32)` | Takes only the first N elements. |

