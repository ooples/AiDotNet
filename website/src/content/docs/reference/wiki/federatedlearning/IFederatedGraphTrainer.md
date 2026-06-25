---
title: "IFederatedGraphTrainer<T>"
description: "Orchestrates federated learning across clients holding subgraphs of a larger graph."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Graph`

Orchestrates federated learning across clients holding subgraphs of a larger graph.

## For Beginners

Standard FL assumes clients have independent datasets. Graph FL is different
because clients' data is interconnected — edges may cross client boundaries. This trainer handles
the unique challenges of graph FL:

## Properties

| Property | Summary |
|:-----|:--------|
| `ClientCount` | Gets the number of registered clients. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverCrossClientEdges` | Initiates cross-client edge discovery between all client pairs. |
| `ExecuteRound(Int32)` | Executes one round of federated graph learning. |
| `GetGlobalModel` | Gets the current global GNN model parameters. |
| `RegisterSubgraph(Int32,Tensor<>,Tensor<>,Tensor<>)` | Registers a client's subgraph with the trainer. |

