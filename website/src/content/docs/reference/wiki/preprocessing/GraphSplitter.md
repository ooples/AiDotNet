---
title: "GraphSplitter<T>"
description: "Graph splitter for node-level or edge-level predictions on graph data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific`

Graph splitter for node-level or edge-level predictions on graph data.

## For Beginners

Graph data consists of nodes (entities) connected by edges (relationships).
Examples include social networks, molecular structures, and citation networks.

## How It Works

**Split Types:**

- Node split: Nodes are divided into train/test sets
- Edge split: Edges are divided (for link prediction tasks)
- Inductive: Test nodes are completely unseen during training

**When to Use:**

- Social network analysis
- Molecular property prediction
- Knowledge graph completion
- Recommendation systems

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphSplitter(Double,GraphSplitter<>.GraphSplitType,Boolean,Int32)` | Creates a new graph splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |
| `WithAdjacencyMatrix(Int32[0:,0:])` | Sets the adjacency matrix for graph-aware splitting. |

