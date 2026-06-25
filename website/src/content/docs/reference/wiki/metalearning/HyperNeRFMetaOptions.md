---
title: "HyperNeRFMetaOptions<T, TInput, TOutput>"
description: "Configuration options for HyperNeRF Meta-learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for HyperNeRF Meta-learning.

## How It Works

Combines hypernetwork conditioning with NeRF-style positional encoding. Each parameter
index is positionally encoded using sinusoidal frequencies, and combined with a task
latent code to produce per-parameter learning rate modulation. This gives the
hypernetwork structural awareness of parameter positions within the network.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConditioningRegWeight` | L2 regularization weight on the conditioning MLP weights to prevent overfitting the hypernetwork. |
| `ConditioningStrength` | Strength of frequency-based conditioning modulation. |
| `LatentDim` | Dimension of the task-specific latent code derived from gradient statistics. |
| `NumFrequencyBands` | Number of sinusoidal frequency bands for positional encoding of parameter indices. |

