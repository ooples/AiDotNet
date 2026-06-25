---
title: "HyperCLIPAlgorithm<T, TInput, TOutput>"
description: "Implementation of HyperCLIP: Contrastive Learning for Hypernetwork-based Meta-Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of HyperCLIP: Contrastive Learning for Hypernetwork-based Meta-Learning.

## How It Works

HyperCLIP uses contrastive alignment between task embeddings (from support gradients)
and parameter embeddings (from adapted parameter deltas). An InfoNCE contrastive loss
aligns each task's gradient signature with its adapted parameter fingerprint, learning
a shared projection space. This cross-modal alignment improves adaptation quality.

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
| `_projectionWeights` | Projection weights: task projection (embDim × projDim) + param projection (embDim × projDim). |

