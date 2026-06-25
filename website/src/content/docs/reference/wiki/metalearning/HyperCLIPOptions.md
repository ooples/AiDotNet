---
title: "HyperCLIPOptions<T, TInput, TOutput>"
description: "Configuration options for HyperCLIP meta-learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for HyperCLIP meta-learning.

## How It Works

HyperCLIP uses contrastive alignment between task embeddings (from support gradients)
and parameter embeddings (from adapted parameters). An InfoNCE-style contrastive loss
ensures that a task's embedding is closest to its own adapted parameter embedding,
enabling zero-shot parameter generation for new tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `ContrastiveTemperature` | Temperature for the InfoNCE contrastive loss (τ in exp(sim/τ)). |
| `ContrastiveWeight` | Weight of the contrastive alignment loss relative to the task loss. |
| `EmbeddingDim` | Gradient compression dimension for computing embeddings from gradients. |
| `ProjectionDim` | Dimension of the shared projection space for contrastive alignment. |

