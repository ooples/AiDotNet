---
title: "GraphData<T>"
description: "Represents a single graph with nodes, edges, features, and optional labels."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Structures`

Represents a single graph with nodes, edges, features, and optional labels.

## For Beginners

Think of a graph as a social network:

- **Nodes**: People in the network
- **Edges**: Friendships or connections between people
- **Node Features**: Each person's attributes (age, interests, etc.)
- **Edge Features**: Relationship attributes (how long they've been friends, interaction frequency)
- **Labels**: What we want to predict (e.g., will this person like a product?)

This class packages all this information together for graph neural network training.

## How It Works

GraphData encapsulates all information about a graph structure including:

- Node features (attributes for each node)
- Edge indices (connections between nodes)
- Edge features (optional attributes for edges)
- Adjacency matrix (graph structure in matrix form)
- Labels (for supervised learning tasks)

## Properties

| Property | Summary |
|:-----|:--------|
| `AdjacencyMatrix` | Adjacency matrix of shape [num_nodes, num_nodes] or [batch_size, num_nodes, num_nodes]. |
| `EdgeFeatures` | Optional edge feature matrix of shape [num_edges, num_edge_features]. |
| `EdgeIndex` | Edge index tensor of shape [2, num_edges] or [num_edges, 2]. |
| `GraphLabel` | Graph-level label for graph-level tasks (e.g., graph classification). |
| `Metadata` | Metadata for heterogeneous graphs (optional). |
| `NodeFeatures` | Node feature matrix of shape [num_nodes, num_features]. |
| `NodeLabels` | Node labels for node-level tasks (e.g., node classification). |
| `NumEdgeFeatures` | Number of edge features (0 if no edge features). |
| `NumEdges` | Number of edges in the graph. |
| `NumNodeFeatures` | Number of node features. |
| `NumNodes` | Number of nodes in the graph. |
| `TestMask` | Mask indicating which nodes are in the test set. |
| `TrainMask` | Mask indicating which nodes are in the training set. |
| `ValMask` | Mask indicating which nodes are in the validation set. |

