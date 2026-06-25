---
title: "IGraphDataLoader<T>"
description: "Interface for data loaders that provide graph-structured data for graph neural networks."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for data loaders that provide graph-structured data for graph neural networks.

## For Beginners

Graphs represent relationships between things:

**Example: Social Network**

- Nodes: People
- Edges: Friendships
- Node Features: Age, interests, location
- Task: Predict user interests based on their friends

**Example: Molecular Structure**

- Nodes: Atoms
- Edges: Chemical bonds
- Node Features: Atom type, charge
- Task: Predict molecular properties (toxicity, activity)

The adjacency matrix tells the GNN which nodes are connected so it can
aggregate information from neighbors during message passing.

## How It Works

This interface is for loading graph-structured data where:

- Nodes have features (attributes for each entity)
- Edges define connections between nodes
- Labels can be per-node (node classification) or per-graph (graph classification)

## Properties

| Property | Summary |
|:-----|:--------|
| `AdjacencyMatrix` | Gets the adjacency matrix of shape [numNodes, numNodes]. |
| `EdgeIndex` | Gets the edge index tensor in COO format [numEdges, 2]. |
| `GraphLabels` | Gets graph labels for graph classification tasks, or null if not available. |
| `NodeFeatures` | Gets the node feature tensor of shape [numNodes, numFeatures]. |
| `NodeLabels` | Gets node labels for node classification tasks, or null if not available. |
| `NumClasses` | Gets the number of classes for classification tasks. |
| `NumEdges` | Gets the number of edges in the graph (or total across all graphs). |
| `NumGraphs` | Gets the number of graphs in the dataset (1 for single-graph datasets like citation networks). |
| `NumNodeFeatures` | Gets the number of node features. |
| `NumNodes` | Gets the number of nodes in the graph (or total across all graphs). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateGraphClassificationTask(Double,Double,Nullable<Int32>)` | Creates a graph classification task for datasets with multiple graphs. |
| `CreateLinkPredictionTask(Double,Double,Nullable<Int32>)` | Creates a link prediction task for predicting missing edges. |
| `CreateNodeClassificationTask(Double,Double,Nullable<Int32>)` | Creates a node classification task with train/val/test split. |

