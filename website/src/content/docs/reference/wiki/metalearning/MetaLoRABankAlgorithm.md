---
title: "MetaLoRABankAlgorithm<T, TInput, TOutput>"
description: "Implementation of Meta-LoRA Bank (2024)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Meta-LoRA Bank (2024).

## How It Works

Meta-LoRA Bank maintains a bank of K diverse LoRA modules, each representing a
learned adaptation pattern. For a new task, a gating network computes task-conditioned
scores and selects the top-K modules via sparse gating. The selected modules are
combined with learned gating weights to produce the task-specific adaptation.

**Algorithm:**

**Advantages:** Each module specializes in a different type of task adaptation.
The gating mechanism enables compositional generalization — novel tasks can be handled
by new combinations of existing modules.

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
| `_gatingParams` | Gating network parameters: maps task embeddings to module scores. |
| `_moduleBank` | Bank of LoRA modules. |

