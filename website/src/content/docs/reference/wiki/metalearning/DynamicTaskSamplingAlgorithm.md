---
title: "DynamicTaskSamplingAlgorithm<T, TInput, TOutput>"
description: "Implementation of Dynamic Task Sampling: difficulty-aware gradient reweighting for meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Dynamic Task Sampling: difficulty-aware gradient reweighting for meta-learning.

## How It Works

Dynamic Task Sampling maintains running difficulty estimates for tasks in the meta-batch
and uses difficulty-proportional weighting on meta-gradients. Tasks with higher post-adaptation
query loss (= harder tasks) receive higher gradient weights, focusing meta-learning
on areas where the model struggles most. An exploration bonus (UCB-style) ensures that
tasks seen less frequently still receive gradient signal.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_taskDifficulty` | Running difficulty estimate per task slot in meta-batch. |
| `_taskVisits` | Visit count per task slot (for UCB exploration). |
| `_totalIterations` | Total meta-iteration count. |

