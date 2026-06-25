---
title: "CausalVAEAlgorithm<T>"
description: "CausalVAE — Causal Variational Autoencoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.DeepLearning`

CausalVAE — Causal Variational Autoencoder.

## For Beginners

CausalVAE learns a compressed version of your data where the
compressed variables have causal relationships between them. This is useful for
understanding underlying causal mechanisms even in high-dimensional data like images.

## How It Works

CausalVAE extends the VAE framework to learn a disentangled latent space where the
latent variables are causally related according to a learned DAG. The encoder maps
observed data to latent exogenous noise, the causal layer transforms independent noise
into causally-structured latent variables via a learned adjacency matrix A, and the
decoder reconstructs the observed data.

**Algorithm:**

- Encoder: X → (mu_eps, logvar_eps) via MLP, sample epsilon ~ N(mu, sigma^2)
- Causal layer: Z = (I - A)^{-1} * epsilon, where A is learned adjacency
- Decoder: Z → X_hat via MLP
- Loss = reconstruction + KL(q(eps)||p(eps)) + sparsity(A) + acyclicity(A)
- A is parameterized via sigmoid of learnable logits with NOTEARS constraint
- Threshold final A to get DAG

Reference: Yang et al. (2021), "CausalVAE: Disentangled Representation Learning via
Neural Structural Causal Models", CVPR.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

