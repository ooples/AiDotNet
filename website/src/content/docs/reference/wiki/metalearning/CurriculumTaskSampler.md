---
title: "CurriculumTaskSampler<T, TInput, TOutput>"
description: "Samples tasks following a difficulty-based curriculum: starts with easy tasks and gradually increases difficulty as training progresses."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

Samples tasks following a difficulty-based curriculum: starts with easy tasks and
gradually increases difficulty as training progresses. Uses episode difficulty scores
and observed losses to order task presentation.

## For Beginners

Just like teaching a student easy problems before hard ones,
curriculum learning presents the meta-learner with simple tasks first and gradually
introduces harder tasks. This typically leads to faster and more stable training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CurriculumTaskSampler(IMetaDataset<,,>,Int32,Int32,Int32,Double,Double,Nullable<Int32>)` | Creates a curriculum task sampler that increases difficulty over time. |

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
| `CurriculumCandidates` | Number of candidates to sample for curriculum-based selection. |

