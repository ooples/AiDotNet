---
title: "TaskCondHyperNetAlgorithm<T, TInput, TOutput>"
description: "Implementation of Task-Conditioned HyperNetwork for meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Task-Conditioned HyperNetwork for meta-learning.

## How It Works

A hypernetwork generates task-specific parameter deltas conditioned on a task embedding
derived from support-set gradient statistics. The chunked hypernetwork architecture
processes the task embedding through a shared hidden layer, then uses per-chunk output
heads to produce parameter deltas for each chunk of the target network.

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
| `_hyperNetWeights` | Hypernetwork weights: W_h (embDim × hiddenDim) + b_h (hiddenDim) + per-chunk W_c (hiddenDim × chunkSize). |

