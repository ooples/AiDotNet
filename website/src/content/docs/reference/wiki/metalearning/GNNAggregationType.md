---
title: "GNNAggregationType"
description: "Specifies how nodes are aggregated to form graph-level representations."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Options`

Specifies how nodes are aggregated to form graph-level representations.

## Fields

| Field | Summary |
|:-----|:--------|
| `Attention` | Attention-weighted aggregation of node embeddings. |
| `Max` | Maximum over each dimension of node embeddings. |
| `Mean` | Mean pooling of all node embeddings. |
| `Set2Set` | Set2Set aggregation using LSTM. |
| `Sum` | Sum of all node embeddings. |

