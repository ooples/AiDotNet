---
title: "HeterogeneousGraphMetadata"
description: "Represents metadata for heterogeneous graphs with multiple node and edge types."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents metadata for heterogeneous graphs with multiple node and edge types.

## For Beginners

This defines the "schema" of your heterogeneous graph.

Think of a knowledge graph with different types of entities and relationships:

- Node types: Person, Company, Product
- Edge types: WorksAt, Manufactures, Purchases

This metadata tells the layer what types exist and how they connect.

## Properties

| Property | Summary |
|:-----|:--------|
| `EdgeTypeSchema` | Edge type connections: maps edge type to (source node type, target node type). |
| `EdgeTypes` | Names of edge types (e.g., ["likes", "belongs_to", "similar_to"]). |
| `NodeTypeFeatures` | Input feature dimensions for each node type. |
| `NodeTypes` | Names of node types (e.g., ["user", "item", "category"]). |

