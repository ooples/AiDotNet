---
title: "MemoryGraphStore<T>"
description: "In-memory implementation of `IGraphStore` using dictionaries for fast lookups."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

In-memory implementation of `IGraphStore` using dictionaries for fast lookups.

## For Beginners

This stores your graph in the computer's memory (RAM).

Pros:

- Very fast (everything in RAM)
- Simple to use (no setup required)

Cons:

- Data lost when app closes
- Limited by available RAM
- Not thread-safe (single-threaded use only)

Good for:

- Development and testing
- Small to medium graphs (<100K nodes)
- Temporary graphs that don't need persistence

Not good for:

- Production systems requiring persistence
- Very large graphs (>1M nodes)
- Multi-process or multi-threaded access to the same graph

For persistent storage, use FileGraphStore or Neo4jGraphStore instead.

## How It Works

This implementation provides high-performance graph storage entirely in RAM.
All operations are O(1) or O(degree) complexity. Data is lost when the application stops.

**Thread Safety:** This class is NOT thread-safe. Callers must ensure proper
synchronization when accessing from multiple threads. For thread-safe operations,
use external locking or consider using `FileGraphStore` which provides
thread-safe access via ConcurrentDictionary.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemoryGraphStore` | Initializes a new instance of the `MemoryGraphStore` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EdgeCount` |  |
| `NodeCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddEdge(GraphEdge<>)` |  |
| `AddEdgeAsync(GraphEdge<>)` |  |
| `AddNode(GraphNode<>)` |  |
| `AddNodeAsync(GraphNode<>)` |  |
| `Clear` |  |
| `ClearAsync` |  |
| `GetAllEdges` |  |
| `GetAllEdgesAsync` |  |
| `GetAllNodes` |  |
| `GetAllNodesAsync` |  |
| `GetEdge(String)` |  |
| `GetEdgeAsync(String)` |  |
| `GetIncomingEdges(String)` |  |
| `GetIncomingEdgesAsync(String)` |  |
| `GetNode(String)` |  |
| `GetNodeAsync(String)` |  |
| `GetNodesByLabel(String)` |  |
| `GetNodesByLabelAsync(String)` |  |
| `GetOutgoingEdges(String)` |  |
| `GetOutgoingEdgesAsync(String)` |  |
| `RemoveEdge(String)` |  |
| `RemoveEdgeAsync(String)` |  |
| `RemoveNode(String)` |  |
| `RemoveNodeAsync(String)` |  |

