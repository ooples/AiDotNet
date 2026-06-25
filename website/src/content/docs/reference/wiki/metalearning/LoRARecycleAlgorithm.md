---
title: "LoRARecycleAlgorithm<T, TInput, TOutput>"
description: "Implementation of LoRA-Recycle (Hu et al., CVPR 2025)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of LoRA-Recycle (Hu et al., CVPR 2025).

## How It Works

LoRA-Recycle enables tuning-free few-shot adaptation by recycling pre-tuned LoRA adapters.
It maintains a bank of task-specific LoRA adapters and a prototype-based selection mechanism
that computes task embeddings from the support set to weight and combine stored adapters
without any gradient-based inner-loop optimization.

**Algorithm:**

**Key difference from MAML:** No gradient-based inner-loop optimization at
inference time. Adaptation is a single forward pass through the prototype encoder
followed by adapter selection and fusion.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `AddToBank(Vector<>,Vector<>)` | Adds an adapter and its prototype to the circular bank. |
| `ComputeTaskEmbedding()` | Computes a task embedding by running support data through the model and projecting the feature representation through the prototype encoder. |
| `FuseAdapters(Vector<>)` | Fuses adapters from the bank using softmax-weighted combination based on similarity between the task embedding and adapter prototypes. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adapterBank` | Bank of LoRA adapters. |
| `_adapterPrototypes` | Prototype embedding for each adapter in the bank (length = prototypeDim per adapter). |
| `_encoderParams` | Prototype encoder parameters: maps feature vectors to prototype space. |

