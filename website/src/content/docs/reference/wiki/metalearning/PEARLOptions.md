---
title: "PEARLOptions<T, TInput, TOutput>"
description: "Configuration options for PEARL: Probabilistic Embeddings for Actor-critic RL (Rakelly et al., ICML 2019)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for PEARL: Probabilistic Embeddings for Actor-critic RL
(Rakelly et al., ICML 2019).

## How It Works

PEARL uses a probabilistic context encoder to infer a latent task variable z from
transition data. The context encoder produces a Gaussian posterior q(z|c) that
conditions the policy and value function. Task inference is amortized and
gradient-free at test time — the encoder simply processes new transitions.

## Properties

| Property | Summary |
|:-----|:--------|
| `EncoderHiddenDim` | Context encoder hidden dimension. |
| `KLWeight` | KL divergence weight for the posterior regularization. |
| `LatentDim` | Dimensionality of the latent task variable z. |
| `NumPosteriorSamples` | Number of posterior samples for training. |

