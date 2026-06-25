---
title: "PEARLAlgorithm<T, TInput, TOutput>"
description: "Implementation of PEARL: Probabilistic Embeddings for Actor-critic RL (Rakelly et al., ICML 2019)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of PEARL: Probabilistic Embeddings for Actor-critic RL
(Rakelly et al., ICML 2019).

## How It Works

PEARL infers a latent task variable z from context (support data) using a probabilistic
encoder q(z|c). The encoder produces a Gaussian posterior, and z is sampled via
reparameterization. The policy/model is conditioned on z through parameter modulation.
At test time, z is inferred from support data without gradient updates.

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
| `_encoderParams` | Encoder parameters: maps compressed gradient → (μ, log_σ²) of size 2*latentDim. |
| `_projectionParams` | Projection matrix W_z: maps z (latentDim) → parameter delta (compressedDim). |

