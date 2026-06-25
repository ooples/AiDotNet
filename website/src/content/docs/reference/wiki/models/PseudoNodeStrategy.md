---
title: "PseudoNodeStrategy"
description: "Specifies how to handle missing cross-client neighbor nodes in subgraph-level FL."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies how to handle missing cross-client neighbor nodes in subgraph-level FL.

## For Beginners

When a graph is split across clients, some edges point to nodes
on other clients. A GNN needs neighbor features for message passing, but it can't directly
access nodes on other clients. Pseudo-node strategies fill in these missing neighbors:

## Fields

| Field | Summary |
|:-----|:--------|
| `FeatureAverage` | Use average feature vectors for missing neighbors. |
| `GeneratorBased` | Use a trained generator to produce pseudo-node features. |
| `None` | Ignore missing cross-client neighbors. |
| `ZeroFill` | Fill missing neighbors with zero vectors. |

