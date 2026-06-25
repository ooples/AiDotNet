---
title: "GraphFLMode"
description: "Specifies the federated graph learning task type."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the federated graph learning task type.

## For Beginners

Graphs can be analyzed at different levels of granularity.
This enum tells the system what kind of graph task each client is performing:

## Fields

| Field | Summary |
|:-----|:--------|
| `GraphClassification` | Each client classifies complete graphs (e.g., molecular property prediction). |
| `LinkPrediction` | Federated link prediction (edge existence). |
| `NodeLevel` | Node-level classification across federated subgraphs. |
| `SubgraphLevel` | Each client holds a subgraph of a larger graph. |

