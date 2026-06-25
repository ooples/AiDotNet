---
title: "iTAMLAlgorithm<T, TInput, TOutput>"
description: "Implementation of iTAML: incremental Task-Agnostic Meta-Learning (Rajasegaran et al., 2020)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of iTAML: incremental Task-Agnostic Meta-Learning (Rajasegaran et al., 2020).

## How It Works

iTAML prevents catastrophic forgetting by maintaining an exponential moving average (EMA)
teacher model. The meta-objective combines task-specific loss with a knowledge distillation
loss that preserves the teacher's predictions. Task-balanced gradient weighting normalizes
gradient magnitudes across tasks to prevent any single task from dominating.

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
| `_teacherParams` | Teacher model parameters (EMA of student). |

