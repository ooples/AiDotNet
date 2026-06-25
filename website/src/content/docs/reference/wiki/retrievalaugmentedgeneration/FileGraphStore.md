---
title: "FileGraphStore<T>"
description: "File-based implementation of `IGraphStore` with persistent storage on disk."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

File-based implementation of `IGraphStore` with persistent storage on disk.

## For Beginners

This stores your graph on disk so it survives restarts.

How it works:

1. When you add a node, it's written to nodes.dat
2. The position (offset) is recorded in node_index.db
3. To retrieve a node, we look up its offset and read from that position
4. Everything is saved to disk automatically

Pros:

- 💾 Data persists across restarts
- 🔄 Can handle graphs larger than RAM
- 📁 Simple file-based storage (no database required)

Cons:

- 🐌 Slower than in-memory (disk I/O overhead)
- 🔒 Not suitable for concurrent access from multiple processes
- 📦 No compression (files can be large)

Good for:

- Applications that need to save graph state
- Graphs up to a few million nodes
- Single-process applications

Not good for:

- Real-time systems requiring sub-millisecond latency
- Multi-process concurrent access
- Distributed systems (use Neo4j or similar instead)

## How It Works

This implementation provides persistent graph storage using files:

- nodes.dat: Binary file containing serialized nodes
- edges.dat: Binary file containing serialized edges
- node_index.db: B-Tree index mapping node IDs to file offsets
- edge_index.db: B-Tree index mapping edge IDs to file offsets

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FileGraphStore(String,WriteAheadLog)` | Initializes a new instance of the `FileGraphStore` class. |

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
| `Dispose` | Disposes the file graph store, ensuring all changes are flushed to disk. |
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
| `ReadExactly(Stream,Byte[],Int32,Int32)` | Reads exactly the specified number of bytes from the stream. |
| `ReadExactlyAsync(Stream,Byte[],Int32,Int32)` | Asynchronously reads exactly the specified number of bytes from the stream. |
| `RebuildInMemoryIndices` | Rebuilds in-memory indices by scanning all nodes and edges. |
| `RemoveEdge(String)` |  |
| `RemoveEdgeAsync(String)` |  |
| `RemoveNode(String)` |  |
| `RemoveNodeAsync(String)` |  |

