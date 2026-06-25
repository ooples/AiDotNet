---
title: "BatchEpisodeSampler<T, TInput, TOutput>"
description: "Efficiently samples batches of episodes with optional prefetching and caching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

Efficiently samples batches of episodes with optional prefetching and caching.
Wraps an `ITaskSampler` and provides batch-level operations
such as prefetching the next batch while the current batch is being trained on.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BatchEpisodeSampler(ITaskSampler<,,>,Int32,Int32)` | Creates a batch episode sampler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets the number of episodes per batch. |
| `PrefetchedCount` | Gets the number of episodes currently in the prefetch buffer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `NextBatch` | Gets the next batch of episodes. |
| `NextTaskBatch` | Gets a task batch suitable for `TaskBatch{`. |
| `ProvideFeedback(IReadOnlyList<IEpisode<,,>>,IReadOnlyList<Double>)` | Provides feedback for the sampled episodes (delegates to underlying sampler). |

