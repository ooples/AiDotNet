---
title: "Episode<T, TInput, TOutput>"
description: "Concrete implementation of `IEpisode` that wraps a meta-learning task with episode-level metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

Concrete implementation of `IEpisode` that wraps a
meta-learning task with episode-level metadata.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Episode(IMetaLearningTask<,,>,String,Nullable<Double>,Dictionary<String,Object>)` | Creates a new episode wrapping the given task. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CreatedTimestamp` |  |
| `Difficulty` |  |
| `Domain` |  |
| `EpisodeId` |  |
| `EpisodeMetadata` |  |
| `LastLoss` |  |
| `SampleCount` |  |
| `Task` |  |

