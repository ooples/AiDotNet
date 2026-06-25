---
title: "IGraphAggregationStrategy<T>"
description: "Graph-aware model aggregation strategy for federated GNN training."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Graph`

Graph-aware model aggregation strategy for federated GNN training.

## For Beginners

Standard FedAvg treats all model parameters equally during aggregation.
But GNNs have distinct parameter types — message-passing weights, attention heads, node embeddings,
readout layers — that benefit from different aggregation treatment.

## How It Works

**Example:** Message-passing layers should be weighted by subgraph size (number of edges),
while readout layers should be weighted by number of labeled nodes. A graph-aware strategy handles
this distinction.

## Properties

| Property | Summary |
|:-----|:--------|
| `StrategyName` | Gets the name of this graph aggregation strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Tensor<>>,Dictionary<Int32,ClientGraphStats>)` | Aggregates GNN model parameters from multiple clients with graph-aware weighting. |

