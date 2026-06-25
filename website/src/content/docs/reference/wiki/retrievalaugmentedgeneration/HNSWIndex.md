---
title: "HNSWIndex<T>"
description: "Hierarchical Navigable Small World (HNSW) graph-based index for approximate nearest neighbor search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes`

Hierarchical Navigable Small World (HNSW) graph-based index for approximate nearest neighbor search.

## How It Works

HNSW builds a multi-layer graph structure where each layer is a proximity graph.
Search starts at the top layer and progressively refines results by moving down layers.
This provides excellent recall with logarithmic search complexity.

Search complexity: O(log n) on average where n is the number of vectors.
Best for large datasets (100K+ vectors) requiring high recall and fast search.

Based on the paper: "Efficient and robust approximate nearest neighbor search using
Hierarchical Navigable Small World graphs" by Malkov and Yashunin (2018).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HNSWIndex(ISimilarityMetric<>,Int32,Int32,Int32,Int32)` | Initializes a new instance of the HNSWIndex class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(String,Vector<>)` |  |
| `AddBatch(Dictionary<String,Vector<>>)` |  |
| `AddBidirectionalEdge(String,String,Int32,Int32)` | Adds a bidirectional edge between two nodes at a specific level. |
| `AddReverseEdgeAndPrune(String,String,Int32,Int32)` | Adds reverse edge from neighbor to node and prunes if needed. |
| `Clear` |  |
| `ConnectNodeAtAllLevels(String,Vector<>,Int32,String)` | Connects a new node to neighbors at all applicable levels. |
| `ConnectNodeToNeighbors(String,List<ValueTuple<String,>>,Int32,Int32)` | Creates bidirectional connections between a node and its neighbors at a specific level. |
| `FindInsertionEntryPoint(Vector<>,Int32)` | Finds the entry point for inserting a new node by traversing from top level. |
| `GetRandomLevel` | Generates a random level for a new node using exponential distribution. |
| `GreedySearchClosest(Vector<>,String,Int32)` | Performs greedy search to find the single closest node at a given level. |
| `InitializeNode(String)` | Initializes a new node in the graph with random level assignment. |
| `InsertSorted(List<ValueTuple<String,>>,ValueTuple<String,>)` | Inserts an item into a sorted list maintaining sort order. |
| `IsBetterScore(,)` | Returns true if score a is better than score b according to the metric. |
| `PruneConnections(String,Int32,Int32)` | Prunes connections for a node to keep only the best ones. |
| `Remove(String)` |  |
| `Search(Vector<>,Int32)` |  |
| `SearchLayer(Vector<>,String,Int32,Int32)` | Searches a single layer for ef nearest neighbors using beam search. |
| `SelectNeighbors(Vector<>,List<ValueTuple<String,>>,Int32)` | Selects the best neighbors from candidates using simple selection. |

