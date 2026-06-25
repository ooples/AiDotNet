---
title: "IGraphStore<T>"
description: "Defines the contract for graph storage backends that manage nodes and edges."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for graph storage backends that manage nodes and edges.

## For Beginners

A graph store is like a filing system for connected information.

Think of it like organizing a network of friends:

- Nodes are people (Alice, Bob, Charlie)
- Edges are relationships (Alice KNOWS Bob, Bob WORKS_WITH Charlie)
- The graph store remembers all these connections

Different implementations might:

- MemoryGraphStore: Keep everything in RAM (fast but lost when app closes)
- FileGraphStore: Save to disk (slower but survives restarts)
- Neo4jGraphStore: Use a professional graph database (production-scale)

This interface lets you swap storage backends without changing your code!

## How It Works

A graph store provides persistent or in-memory storage for knowledge graphs,
enabling efficient storage and retrieval of entities (nodes) and their relationships (edges).
Implementations can range from simple in-memory dictionaries to distributed graph databases.

## Properties

| Property | Summary |
|:-----|:--------|
| `EdgeCount` | Gets the total number of edges in the graph store. |
| `NodeCount` | Gets the total number of nodes in the graph store. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddEdge(GraphEdge<>)` | Adds an edge to the graph representing a relationship between two nodes. |
| `AddEdgeAsync(GraphEdge<>)` | Asynchronously adds an edge to the graph. |
| `AddNode(GraphNode<>)` | Adds a node to the graph or updates it if it already exists. |
| `AddNodeAsync(GraphNode<>)` | Asynchronously adds a node to the graph or updates it if it already exists. |
| `Clear` | Removes all nodes and edges from the graph. |
| `ClearAsync` | Asynchronously removes all nodes and edges from the graph. |
| `GetAllEdges` | Gets all edges currently stored in the graph. |
| `GetAllEdgesAsync` | Asynchronously gets all edges currently stored in the graph. |
| `GetAllNodes` | Gets all nodes currently stored in the graph. |
| `GetAllNodesAsync` | Asynchronously gets all nodes currently stored in the graph. |
| `GetEdge(String)` | Retrieves an edge by its unique identifier. |
| `GetEdgeAsync(String)` | Asynchronously retrieves an edge by its unique identifier. |
| `GetIncomingEdges(String)` | Gets all incoming edges to a specific node. |
| `GetIncomingEdgesAsync(String)` | Asynchronously gets all incoming edges to a specific node. |
| `GetNode(String)` | Retrieves a node by its unique identifier. |
| `GetNodeAsync(String)` | Asynchronously retrieves a node by its unique identifier. |
| `GetNodesByLabel(String)` | Gets all nodes with a specific label. |
| `GetNodesByLabelAsync(String)` | Asynchronously gets all nodes with a specific label. |
| `GetOutgoingEdges(String)` | Gets all outgoing edges from a specific node. |
| `GetOutgoingEdgesAsync(String)` | Asynchronously gets all outgoing edges from a specific node. |
| `RemoveEdge(String)` | Removes an edge from the graph. |
| `RemoveEdgeAsync(String)` | Asynchronously removes an edge from the graph. |
| `RemoveNode(String)` | Removes a node and all its connected edges from the graph. |
| `RemoveNodeAsync(String)` | Asynchronously removes a node and all its connected edges from the graph. |

