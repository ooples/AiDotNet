---
title: "BalancedTaskSampler<T, TInput, TOutput>"
description: "Samples tasks while ensuring that all classes in the meta-dataset appear equally often across the sampled episodes over time."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

Samples tasks while ensuring that all classes in the meta-dataset appear equally often
across the sampled episodes over time. This prevents the meta-learner from overfitting to
frequently-sampled classes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BalancedTaskSampler(IMetaDataset<,,>,Int32,Int32,Int32,Nullable<Int32>)` | Creates a balanced task sampler that rotates through all classes evenly. |

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

## Fields

| Field | Summary |
|:-----|:--------|
| `BalanceCandidates` | Number of candidates to sample for class-balanced selection. |

