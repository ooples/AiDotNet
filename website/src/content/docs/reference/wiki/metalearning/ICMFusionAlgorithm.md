---
title: "ICMFusionAlgorithm<T, TInput, TOutput>"
description: "Implementation of ICM-Fusion: In-Context Meta-Optimized LoRA Fusion (2025)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of ICM-Fusion: In-Context Meta-Optimized LoRA Fusion (2025).

## How It Works

ICM-Fusion addresses the problem of fusing multiple task-specific parameter deltas (task vectors)
by encoding them into a shared latent space via a Fusion-VAE. The VAE learns a manifold where
task vector arithmetic resolves inter-weight conflicts that arise when naively averaging or
summing adapters from different tasks.

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
| `_componentLatents` | Stored latent codes from recent tasks for fusion (circular buffer). |
| `_decoderParams` | VAE decoder params: maps latent vectors back to parameter deltas. |
| `_encoderParams` | VAE encoder params: maps task vectors to (μ, log_σ²) in latent space. |

