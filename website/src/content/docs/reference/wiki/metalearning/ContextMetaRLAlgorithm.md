---
title: "ContextMetaRLAlgorithm<T, TInput, TOutput>"
description: "Implementation of Context Meta-RL: context-conditioned meta-reinforcement learning with multi-head attention-based aggregation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Context Meta-RL: context-conditioned meta-reinforcement learning
with multi-head attention-based aggregation.

## How It Works

Context Meta-RL uses multi-head attention to aggregate task context from the support
set gradient history. A learned query vector attends over encoded gradient entries
to produce a context vector. This context vector multiplicatively modulates the
model parameters, providing a smooth, deterministic task conditioning mechanism.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `AttentionAggregate(List<Vector<>>)` | Multi-head attention aggregation: α_k = softmax(q^T e_k / √d / τ). |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `MaxCompressedDim` | Maximum compressed gradient dimension for encoder input. |
| `_encoderParams` | Context encoder: compressedDim → contextDim. |
| `_modulationParams` | Modulation projection: contextDim → compressedDim. |
| `_queryVector` | Learned attention query vector: contextDim. |

