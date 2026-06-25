---
title: "FedGnnAggregationStrategy<T>"
description: "GNN-aware federated aggregation that weights contributions by subgraph topology characteristics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Graph`

GNN-aware federated aggregation that weights contributions by subgraph topology characteristics.

## For Beginners

Standard FedAvg weights clients by dataset size (number of samples).
For graph FL, "dataset size" is ambiguous — is it nodes, edges, or labeled nodes?
This strategy uses a composite weight based on graph topology:

## How It Works

**Formula:** weight_i = alpha * E_i/E_total + beta * L_i/L_total + gamma * deg_i/deg_avg,
where E = edges, L = labeled nodes, deg = average degree, and alpha+beta+gamma = 1.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedGnnAggregationStrategy(Double,Double,Double)` | Initializes a new instance of `FedGnnAggregationStrategy`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `StrategyName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Tensor<>>,Dictionary<Int32,ClientGraphStats>)` |  |

