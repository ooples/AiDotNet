---
title: "DynamicTaskSampler<T, TInput, TOutput>"
description: "Samples tasks with probability proportional to the loss observed on previous evaluations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

Samples tasks with probability proportional to the loss observed on previous evaluations.
Tasks with higher loss are sampled more frequently, focusing training on areas the model
finds most difficult (inspired by hard-example mining and curriculum learning research).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DynamicTaskSampler(IMetaDataset<,,>,Int32,Int32,Int32,Double,Nullable<Int32>)` | Creates a dynamic task sampler that adapts based on observed losses. |

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
| `SelectionCandidates` | Number of candidates to sample for loss-biased selection. |

