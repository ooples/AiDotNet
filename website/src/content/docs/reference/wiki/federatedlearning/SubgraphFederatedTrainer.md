---
title: "SubgraphFederatedTrainer<T>"
description: "Main coordinator for subgraph-level federated GNN training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Graph`

Main coordinator for subgraph-level federated GNN training.

## For Beginners

This class orchestrates graph FL end-to-end:

## How It Works

**Pseudo-node expansion:** Before local training, each client's subgraph is expanded
with pseudo-nodes that approximate missing cross-client neighbors. The strategy
(FeatureAverage, GeneratorBased, ZeroFill) is controlled by `FederatedGraphOptions`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SubgraphFederatedTrainer(FederatedGraphOptions,IGraphAggregationStrategy<>,ICrossClientEdgeHandler<>)` | Initializes a new instance of `SubgraphFederatedTrainer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClientCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverCrossClientEdges` |  |
| `ExecuteRound(Int32)` |  |
| `GetGlobalModel` |  |
| `RegisterSubgraph(Int32,Tensor<>,Tensor<>,Tensor<>)` |  |

