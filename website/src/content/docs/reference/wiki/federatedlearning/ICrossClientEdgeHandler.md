---
title: "ICrossClientEdgeHandler<T>"
description: "Handles secure discovery and management of edges that cross client boundaries."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Graph`

Handles secure discovery and management of edges that cross client boundaries.

## For Beginners

When a graph is split across clients, some edges connect nodes on
different clients (cross-client edges). Discovering these edges is essential for GNN quality
but must be done privately — Client A shouldn't learn Client B's full adjacency list.

## How It Works

**Approach:** Uses Private Set Intersection (PSI) from issue #538: each client provides
its border node IDs, and the PSI protocol reveals only the intersection (shared nodes) without
exposing non-shared nodes.

## Properties

| Property | Summary |
|:-----|:--------|
| `DiscoveredEdgeCount` | Gets the total number of discovered cross-client edges. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CacheEdges(Int32,Int32,IReadOnlyList<ValueTuple<Int32,Int32>>)` | Caches discovered edges for a client pair (to avoid re-running PSI each round). |
| `DiscoverEdges(IReadOnlyList<Int32>,IReadOnlyList<Int32>)` | Discovers cross-client edges between two clients using PSI. |
| `GetEdges(Int32,Int32)` | Gets the discovered cross-client edges between a specific client pair. |

