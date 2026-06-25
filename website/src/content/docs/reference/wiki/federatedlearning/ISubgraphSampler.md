---
title: "ISubgraphSampler<T>"
description: "Samples neighborhoods from a client's local subgraph for mini-batch GNN training."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Graph`

Samples neighborhoods from a client's local subgraph for mini-batch GNN training.

## For Beginners

GNNs learn by aggregating features from neighboring nodes (message passing).
For large subgraphs, using all neighbors is too expensive. A sampler selects a fixed-size neighborhood
at each hop, creating small "computation trees" for efficient training.

## How It Works

**Example:** For a target node, with 2 hops and max 10 neighbors per hop:

## Methods

| Method | Summary |
|:-----|:--------|
| `Sample(Tensor<>,Int32[],Tensor<>)` | Samples a k-hop neighborhood around the specified target nodes. |

