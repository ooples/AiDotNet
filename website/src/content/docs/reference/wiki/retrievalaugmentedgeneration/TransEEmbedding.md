---
title: "TransEEmbedding<T>"
description: "TransE embedding model: entities and relations are vectors in the same space, with the scoring function d(h, r, t) = ||h + r - t||."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

TransE embedding model: entities and relations are vectors in the same space,
with the scoring function d(h, r, t) = ||h + r - t||.

## For Beginners

Imagine entities as points on a map and relations as directions.
"Paris" + "capital_of" should point to "France". If the model learns good vectors,
you can predict that "Berlin" + "capital_of" ≈ "Germany".

## How It Works

TransE (Bordes et al., 2013) models relations as translations in embedding space.
For a valid triple (h, r, t), the model learns embeddings such that h + r ≈ t.
Training uses margin-based ranking loss: max(0, margin + d_pos - d_neg).

## Properties

| Property | Summary |
|:-----|:--------|
| `IsDistanceBased` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `OnPostEpoch(Int32)` | After each epoch, normalize entity embeddings to the unit ball (TransE constraint). |

