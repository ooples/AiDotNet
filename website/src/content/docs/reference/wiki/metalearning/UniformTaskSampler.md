---
title: "UniformTaskSampler<T, TInput, TOutput>"
description: "Samples tasks uniformly at random from a meta-dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

Samples tasks uniformly at random from a meta-dataset.
This is the simplest and most commonly used sampling strategy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniformTaskSampler(IMetaDataset<,,>,Int32,Int32,Int32)` | Creates a uniform task sampler over the given meta-dataset. |

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

