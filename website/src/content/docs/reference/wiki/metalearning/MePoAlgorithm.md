---
title: "MePoAlgorithm<T, TInput, TOutput>"
description: "Implementation of MePo: Memory Prototypes for continual few-shot meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of MePo: Memory Prototypes for continual few-shot meta-learning.

## How It Works

MePo maintains a memory bank of gradient-space prototypes from previously encountered tasks.
When adapting to a new task, it compresses the initial gradient into the prototype space,
retrieves the K nearest prototypes, and uses their weighted average to regularize
the adaptation trajectory. This prevents catastrophic forgetting by anchoring adaptation
to known good trajectories, while still allowing task-specific fine-tuning.

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
| `_memoryBank` | Memory bank: list of prototype vectors. |

