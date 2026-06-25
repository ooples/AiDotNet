---
title: "HyperNetMetaRLAlgorithm<T, TInput, TOutput>"
description: "Implementation of HyperNet Meta-RL: hypernetwork-based task-specific parameter generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of HyperNet Meta-RL: hypernetwork-based task-specific parameter generation.

## How It Works

A hypernetwork takes a task embedding (computed from the support gradient) and generates
the full policy parameter vector in a single forward pass. The task embedding is computed
by compressing the initial support gradient through a learned encoder. The hypernetwork
is a 2-layer MLP that maps the embedding to a parameter delta vector.

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
| `SpsaLearningRateMultiplier` | SPSA learning rate multiplier for auxiliary parameter updates. |
| `_encoderParams` | Task encoder: compressedDim → embDim. |
| `_hyperLayer1` | Hypernetwork layer 1: embDim → hiddenDim. |
| `_hyperLayer2` | Hypernetwork layer 2: hiddenDim → compressedDim. |

