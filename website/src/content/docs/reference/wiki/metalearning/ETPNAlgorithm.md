---
title: "ETPNAlgorithm<T, TInput, TOutput>"
description: "Implementation of ETPN: Embedding-Transformed Prototypical Networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of ETPN: Embedding-Transformed Prototypical Networks.

## How It Works

ETPN learns a task-specific embedding transformation applied transductively. The
transformation is computed from both support and query gradient information, enabling
the adapted parameter space to be more discriminative for the specific task.
A learned projection maps combined gradient features to per-parameter scaling factors.

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
| `_transformParams` | Transform projection: compressedDim × transformDim. |

