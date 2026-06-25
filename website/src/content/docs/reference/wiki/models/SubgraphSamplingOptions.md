---
title: "SubgraphSamplingOptions"
description: "Configuration for subgraph neighborhood sampling during federated GNN training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for subgraph neighborhood sampling during federated GNN training.

## For Beginners

GNNs learn by aggregating features from neighboring nodes (message passing).
In a large graph, using all neighbors at every hop is too expensive. Neighborhood sampling limits
how many neighbors are used at each hop, trading off accuracy for speed.

## How It Works

**Example:** With HopCount=2 and MaxNeighborsPerHop=10, for each node we sample up to 10
neighbors at hop 1, and for each of those, sample up to 10 neighbors at hop 2. This gives a
2-hop subgraph of up to ~110 nodes instead of potentially thousands.

## Properties

| Property | Summary |
|:-----|:--------|
| `HopCount` | Gets or sets the number of hops for neighborhood sampling. |
| `IncludeSelfLoops` | Gets or sets whether to include self-loops in the sampled subgraph. |
| `MaxNeighborsPerHop` | Gets or sets the maximum number of neighbors to sample per hop. |
| `MinNodeDegree` | Gets or sets the minimum degree a node must have to be included in sampling. |
| `UseImportanceSampling` | Gets or sets whether to use importance sampling (degree-proportional) instead of uniform. |

