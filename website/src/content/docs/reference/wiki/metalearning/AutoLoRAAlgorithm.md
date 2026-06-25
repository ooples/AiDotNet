---
title: "AutoLoRAAlgorithm<T, TInput, TOutput>"
description: "Implementation of AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning (Zhang et al., NAACL 2024)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation
Based on Meta Learning (Zhang et al., NAACL 2024).

## How It Works

AutoLoRA addresses LoRA's limitation of uniform rank assignment across all layers by
using a bi-level optimization to automatically discover the optimal rank for each
parameter group.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ApplyThreshold(Double[][])` | Applies threshold to selection weights, zeroing out components below threshold. |
| `ComposeAdaptedParams(Vector<>,Double[][])` | Composes adapted parameters by adding weighted rank-1 components to base params. |
| `ComputeSelectionWeights` | Computes softmax selection weights for each (group, rank) pair. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `UpdateRankComponents(Vector<>,Vector<>,Double[][])` | Updates rank-1 components using projected gradients from the full parameter gradient. |

## Fields

| Field | Summary |
|:-----|:--------|
| `SpsaLearningRateMultiplier` | SPSA learning rate multiplier for auxiliary parameter updates. |
| `_rankComponents` | Rank-1 components for all groups. |
| `_selectionLogits` | Selection logits β for each (group, component) pair. |

