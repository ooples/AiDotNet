---
title: "RetrievalContext<T>"
description: "Context returned by the retrieval module."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Context returned by the retrieval module.

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionWeights` | Attention weights over neighbors [batchSize, numNeighbors]. |
| `Labels` | Retrieved labels [batchSize, numNeighbors, labelDim]. |
| `NumNeighbors` | Number of neighbors retrieved. |
| `Values` | Retrieved value embeddings [batchSize, numNeighbors, embeddingDim]. |

