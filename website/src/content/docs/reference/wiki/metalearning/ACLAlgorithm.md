---
title: "ACLAlgorithm<T, TInput, TOutput>"
description: "Implementation of ACL: Adaptive Continual Learning with task-specific parameter importance masks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of ACL: Adaptive Continual Learning with task-specific parameter importance masks.

## How It Works

ACL prevents catastrophic forgetting by maintaining per-parameter importance scores that
accumulate across tasks via exponential moving average. Important parameters are protected
by reducing their effective learning rate and applying elastic weight consolidation (EWC)-style
regularization toward the pre-task initialization.

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
| `_importance` | Per-parameter importance scores (accumulated via EMA). |

