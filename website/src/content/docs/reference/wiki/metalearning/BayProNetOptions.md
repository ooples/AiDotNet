---
title: "BayProNetOptions<T, TInput, TOutput>"
description: "Configuration options for BayProNet: Bayesian Prototypical Networks for few-shot learning with uncertainty estimation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for BayProNet: Bayesian Prototypical Networks for few-shot learning
with uncertainty estimation.

## How It Works

BayProNet extends Prototypical Networks by modeling class prototypes as Gaussian
distributions N(μ_c, σ²_c) rather than point estimates. The prototype mean and
variance are computed from support set embeddings, and classification uses the
expected negative log-likelihood under the prototype distribution rather than
simple Euclidean distance.

**Key equations:**

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDim` | Dimensionality of prototype embeddings. |
| `InitialPrototypeLogVar` | Initial log-variance for prototype distributions. |
| `KLWeight` | Weight for the KL divergence between prototype posteriors and a unit Gaussian prior. |
| `Temperature` | Temperature scaling for the softmax over Mahalanobis distances. |

