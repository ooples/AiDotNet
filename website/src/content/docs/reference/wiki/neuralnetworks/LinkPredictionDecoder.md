---
title: "LinkPredictionDecoder<T>"
description: "Decoder types for combining node embeddings into edge scores."
section: "API Reference"
---

`Enums` · `AiDotNet.NeuralNetworks.Tasks.Graph`

Decoder types for combining node embeddings into edge scores.

## Fields

| Field | Summary |
|:-----|:--------|
| `CosineSimilarity` | Cosine similarity: score = (z_i * z_j) / (\|\|z_i\|\| \|\|z_j\|\|) |
| `Distance` | L2 distance: score = -\|\|z_i - z_j\|\|^2 |
| `DotProduct` | Dot product: score = z_i * z_j |
| `Hadamard` | Element-wise product: score = sum(z_i * z_j) |

