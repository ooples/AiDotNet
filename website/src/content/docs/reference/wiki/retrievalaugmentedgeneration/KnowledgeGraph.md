---
title: "KnowledgeGraph<T>"
description: "Knowledge graph for storing and querying entity relationships using a pluggable storage backend."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Knowledge graph for storing and querying entity relationships using a pluggable storage backend.

## For Beginners

A knowledge graph is like a map of how information connects together.

Imagine Wikipedia as a graph:

- Each article is a node (Albert Einstein, Physics, Germany, etc.)
- Links between articles are edges (Einstein STUDIED Physics, Einstein BORN_IN Germany)
- You can traverse the graph to find related information

This class lets you:

1. Add entities and relationships
2. Find connections between entities
3. Traverse the graph to discover related information
4. Query based on entity types or relationships

For example, to answer "Who worked at Princeton?":

1. Find all edges with type "WORKED_AT"
2. Filter for target = "Princeton University"
3. Return the source entities (people who worked there)

Storage backends you can use:

- MemoryGraphStore: Fast, in-memory (default)
- FileGraphStore: Persistent, disk-based
- Neo4jGraphStore: Professional graph database (future)

## How It Works

A knowledge graph stores entities (nodes) and their relationships (edges) to enable structured information retrieval.
This implementation delegates storage operations to an `IGraphStore` implementation,
allowing you to swap between in-memory, file-based, or database-backed storage.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KnowledgeGraph` | Initializes a new instance of the `KnowledgeGraph` class with default in-memory storage. |
| `KnowledgeGraph(IGraphStore<>)` | Initializes a new instance of the `KnowledgeGraph` class with a custom graph store. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EdgeCount` | Gets the total number of edges in the graph. |
| `NodeCount` | Gets the total number of nodes in the graph. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddEdge(GraphEdge<>)` | Adds an edge to the graph. |
| `AddNode(GraphNode<>)` | Adds a node to the graph or updates it if it already exists. |
| `BreadthFirstTraversal(String,Int32)` | Performs breadth-first search traversal starting from a node. |
| `Clear` | Clears all nodes and edges from the graph. |
| `FindRelatedNodes(String,Int32)` | Finds nodes related to a query by entity name or property matching. |
| `FindShortestPath(String,String)` | Finds the shortest path between two nodes using BFS. |
| `GetAllEdges` | Gets all edges in the graph. |
| `GetAllNodes` | Gets all nodes in the graph. |
| `GetEdgesAt(DateTime)` | Gets all edges valid at a specific point in time. |
| `GetIncomingEdges(String)` | Gets all incoming edges to a node. |
| `GetIncomingEdgesAt(String,DateTime)` | Gets incoming edges to a node that are valid at a specific point in time. |
| `GetNeighbors(String)` | Gets all neighbors of a node (nodes connected by outgoing edges). |
| `GetNeighborsAt(String,DateTime)` | Gets neighbors of a node considering only edges valid at a specific point in time. |
| `GetNode(String)` | Gets a node by its ID. |
| `GetNodesByLabel(String)` | Gets all nodes with a specific label. |
| `GetOutgoingEdges(String)` | Gets all outgoing edges from a node. |
| `GetOutgoingEdgesAt(String,DateTime)` | Gets outgoing edges from a node that are valid at a specific point in time. |

