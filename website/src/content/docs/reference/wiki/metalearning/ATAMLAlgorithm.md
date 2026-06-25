---
title: "ATAMLAlgorithm<T, TInput, TOutput>"
description: "Implementation of ATAML: Attention-based Task-Adaptive Meta-Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of ATAML: Attention-based Task-Adaptive Meta-Learning.

## How It Works

ATAML meta-learns a task-adaptive attention mechanism that produces per-parameter
learning rate scaling factors based on the task's gradient profile. A learned projection
maps compressed gradient features to attention weights over parameter dimensions,
enabling the inner loop to focus adaptation on the most relevant parameters for each task.

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
| `_attentionParams` | Attention projection: compressedDim × attentionDim. |

