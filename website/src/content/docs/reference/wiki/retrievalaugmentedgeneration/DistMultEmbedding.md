---
title: "DistMultEmbedding<T>"
description: "DistMult embedding model: bilinear diagonal scoring with Σ(h_k · r_k · t_k)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

DistMult embedding model: bilinear diagonal scoring with Σ(h_k · r_k · t_k).

## For Beginners

DistMult is the simplest bilinear model:

- Each entity and relation gets a vector
- Score = element-wise product of head, relation, and tail vectors, summed up
- Higher score = more likely to be a true fact
- Works best for symmetric relations where direction doesn't matter

## How It Works

DistMult (Yang et al., 2015) uses a diagonal bilinear form to score triples.
Score: Σ_k (h_k · r_k · t_k). Higher scores indicate more plausible triples.
Training uses logistic loss with L2 regularization on embeddings.

DistMult is inherently symmetric: score(h, r, t) = score(t, r, h), making it
best suited for symmetric relations like "similar_to" or "married_to".

## Properties

| Property | Summary |
|:-----|:--------|
| `IsDistanceBased` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `StableSigmoid(Double)` | Numerically stable sigmoid: 1 / (1 + exp(-x)). |
| `StableSoftplus(Double)` | Numerically stable softplus: log(1 + exp(x)). |

