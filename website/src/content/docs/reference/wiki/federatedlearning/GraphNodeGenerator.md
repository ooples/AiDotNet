---
title: "GraphNodeGenerator<T>"
description: "Generates pseudo-node features for missing cross-client neighbors using a learned model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Graph`

Generates pseudo-node features for missing cross-client neighbors using a learned model.

## For Beginners

When a GNN on Client A needs features from a node on Client B (a
cross-client neighbor), it can't directly access them. This generator learns to produce realistic
pseudo-node features based on the local graph structure:

## How It Works

**Training:** The generator is trained locally on each client using known edges —
it learns to predict a node's features from its neighbors' features. Then it's used to predict
features for missing (cross-client) neighbors.

**Homophily assumption:** Works best on homophilic graphs where connected nodes tend to
have similar features (social networks, citation networks). Less effective on heterophilic graphs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphNodeGenerator(Int32,Int32,Double)` | Initializes a new instance of `GraphNodeGenerator`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsTrained` | Gets a value indicating whether the generator has been trained. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Generate(Tensor<>)` | Generates a pseudo-node feature vector for a missing cross-client neighbor. |
| `Train(Tensor<>,Tensor<>,Int32,Int32)` | Trains the generator on known node-neighbor pairs from the local subgraph. |

