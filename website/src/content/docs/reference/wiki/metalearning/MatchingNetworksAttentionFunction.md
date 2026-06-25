---
title: "MatchingNetworksAttentionFunction"
description: "Attention function types for Matching Networks."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Options`

Attention function types for Matching Networks.

## For Beginners

This controls how the network decides which
support examples are most similar to the query example.

## How It Works

The attention function determines how similarity is measured between query
and support embeddings when computing attention weights.

## Fields

| Field | Summary |
|:-----|:--------|
| `Cosine` | Cosine similarity between embeddings. |
| `DotProduct` | Dot product similarity. |
| `Euclidean` | Negative Euclidean distance. |
| `Learned` | Learned similarity function. |

