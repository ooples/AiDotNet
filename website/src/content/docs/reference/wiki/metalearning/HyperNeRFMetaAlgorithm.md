---
title: "HyperNeRFMetaAlgorithm<T, TInput, TOutput>"
description: "Implementation of HyperNeRF Meta: Positional-Encoding-Conditioned Meta-Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of HyperNeRF Meta: Positional-Encoding-Conditioned Meta-Learning.

## How It Works

Combines hypernetwork conditioning with NeRF-style sinusoidal positional encoding.
Each parameter index is encoded using multi-frequency sin/cos functions, providing
structural awareness of where parameters are located in the network. These positional
features are combined with a task-specific latent code (from gradient statistics) to
produce per-parameter learning rate modulation through a learned conditioning MLP.

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
| `_conditioningWeights` | Conditioning MLP weights: (peDim + latentDim) × numGroups. |
| `_positionalEncodings` | Pre-computed positional encodings per group. |

