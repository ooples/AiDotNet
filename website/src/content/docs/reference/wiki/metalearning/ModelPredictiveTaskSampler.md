---
title: "ModelPredictiveTaskSampler<T, TInput, TOutput>"
description: "Model Predictive Task Sampling (MPTS): predicts which tasks will yield the greatest learning signal by maintaining a posterior estimate of per-task adaptation risk, then sampling tasks that balance exploration and exploitation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

Model Predictive Task Sampling (MPTS): predicts which tasks will yield the greatest
learning signal by maintaining a posterior estimate of per-task adaptation risk,
then sampling tasks that balance exploration and exploitation.

## For Beginners

Instead of sampling tasks randomly, MPTS tries to predict
which tasks will help the model learn the most. It keeps track of how much the model
improved on each task and favors tasks where improvement is likely highest.

## How It Works

**Reference:** Model Predictive Task Sampling (2025).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelPredictiveTaskSampler(IMetaDataset<,,>,Int32,Int32,Int32,Double,Double,Nullable<Int32>)` | Creates a model-predictive task sampler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumQueryPerClass` |  |
| `NumShots` |  |
| `NumWays` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeArmKey(IEpisode<,,>)` | Computes a stable arm key for bandit statistics from an episode's domain and structure. |
| `SampleBatch(Int32)` |  |
| `SampleOne` |  |
| `SetSeed(Int32)` |  |
| `UpdateWithFeedback(IReadOnlyList<IEpisode<,,>>,IReadOnlyList<Double>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `ExplorationRate` | Fraction of calls that use pure random exploration. |
| `UcbCandidates` | Number of candidate episodes to sample for UCB selection. |

