---
title: "SubgraphExpander<T>"
description: "Expands a local subgraph with pseudo-nodes to approximate missing cross-client neighbors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Graph`

Expands a local subgraph with pseudo-nodes to approximate missing cross-client neighbors.

## For Beginners

When a graph is split across clients, GNNs on each client "see" only
their local subgraph. Nodes near the boundary are missing their cross-client neighbors, which
degrades message passing. The SubgraphExpander adds "pseudo-nodes" to fill these gaps:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SubgraphExpander(FederatedGraphOptions)` | Initializes a new instance of `SubgraphExpander`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Expand(Tensor<>,Tensor<>,Dictionary<Int32,Tensor<>>)` | Expands a local subgraph with pseudo-nodes for missing cross-client neighbors. |

