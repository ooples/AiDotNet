---
title: "EpisodicDataLoaderTaskSamplerAdapter<T, TInput, TOutput>"
description: "Adapter that wraps an existing `IEpisodicDataLoader` to implement the `ITaskSampler` interface."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

Adapter that wraps an existing `IEpisodicDataLoader`
to implement the `ITaskSampler` interface.
This provides backward compatibility so that legacy data loaders can be used
with the new task-sampler-based infrastructure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EpisodicDataLoaderTaskSamplerAdapter(IEpisodicDataLoader<,,>)` | Creates an adapter wrapping the given episodic data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumQueryPerClass` |  |
| `NumShots` |  |
| `NumWays` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleBatch(Int32)` |  |
| `SampleOne` |  |
| `SetSeed(Int32)` |  |
| `UpdateWithFeedback(IReadOnlyList<IEpisode<,,>>,IReadOnlyList<Double>)` |  |

