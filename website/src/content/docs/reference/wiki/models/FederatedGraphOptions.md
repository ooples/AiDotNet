---
title: "FederatedGraphOptions"
description: "Configuration options for federated graph learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for federated graph learning.

## For Beginners

Graph FL extends standard federated learning to handle graph-structured
data (social networks, molecular graphs, knowledge graphs). Each client holds a subgraph, and
the server coordinates GNN training across all subgraphs. These options control the graph-specific
aspects of training.

## How It Works

**Key decisions:**

## Properties

| Property | Summary |
|:-----|:--------|
| `CrossClientEdges` | Gets or sets cross-client edge handling options. |
| `HiddenDimension` | Gets or sets the hidden layer dimension for GNN layers. |
| `Mode` | Gets or sets the graph FL mode (subgraph, node, link, or graph classification). |
| `NeighborhoodPrivacyEpsilon` | Gets or sets the neighborhood privacy epsilon for LDP on topology queries. |
| `NodeFeatureDimension` | Gets or sets the dimensionality of node feature vectors. |
| `NumGnnLayers` | Gets or sets the number of GNN message-passing layers. |
| `NumberOfPartitions` | Gets or sets the number of target partitions when using graph partitioning. |
| `PartitionStrategy` | Gets or sets the graph partition strategy. |
| `PrototypesPerClass` | Gets or sets the number of prototypes per class for prototype-based learning. |
| `PseudoNodeStrategy` | Gets or sets the pseudo-node strategy for handling missing cross-client neighbors. |
| `Sampling` | Gets or sets subgraph neighborhood sampling options. |
| `UsePrototypeLearning` | Gets or sets whether to use prototype-based learning instead of full model sharing. |

