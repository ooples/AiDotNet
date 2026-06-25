---
title: "VflAggregationMode"
description: "Specifies how embeddings from multiple parties are combined in vertical federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies how embeddings from multiple parties are combined in vertical federated learning.

## For Beginners

In vertical FL, each party computes a local embedding (a compressed
representation of its features). These embeddings must be combined before the top model can
make a prediction. This enum controls how that combination happens.

## Fields

| Field | Summary |
|:-----|:--------|
| `Attention` | Attention-weighted combination of embeddings. |
| `Concatenation` | Concatenate embeddings side by side. |
| `Gating` | Gating mechanism that learns a sigmoid gate to blend embeddings. |
| `Sum` | Element-wise sum of embeddings. |

