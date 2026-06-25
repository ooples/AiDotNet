---
title: "MPTSAlgorithm<T, TInput, TOutput>"
description: "Implementation of MPTS: Meta-learning with Progressive Task-Specific adaptation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of MPTS: Meta-learning with Progressive Task-Specific adaptation.

## How It Works

MPTS divides model parameters into groups and progressively unfreezes them during the
inner loop. High-priority groups (typically the classifier head) are adapted from the
first step, while lower-priority groups (backbone) are gradually unfrozen as adaptation
progresses. Learned priority scores determine the unfreezing schedule.

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
| `_priorityScores` | Learned priority scores for each group (higher = adapted earlier). |

