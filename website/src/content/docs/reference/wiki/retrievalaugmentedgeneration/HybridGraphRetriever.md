---
title: "HybridGraphRetriever<T>"
description: "Hybrid retriever that combines vector similarity search with graph traversal for enhanced RAG."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Hybrid retriever that combines vector similarity search with graph traversal for enhanced RAG.

## For Beginners

Traditional RAG uses only vector similarity:

Query: "What is photosynthesis?"
Traditional RAG:

- Find documents similar to the query
- Return top-k matches
- Misses related context!

Hybrid Graph RAG:

- Find initial matches via vector similarity
- Walk the graph to find related concepts
- Example: photosynthesis → chlorophyll → plants → carbon dioxide
- Provides richer, more complete context

Real-world analogy:

- Traditional: Search "Paris" → get Paris documents
- Hybrid: Search "Paris" → get Paris + France + Eiffel Tower + Seine River
- Graph connections provide context vectors can't capture!

## How It Works

This retriever uses a two-stage approach:

1. Vector similarity search to find initial candidate nodes
2. Graph traversal to expand context with related nodes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HybridGraphRetriever(KnowledgeGraph<>,IDocumentStore<>)` | Initializes a new instance of the `HybridGraphRetriever` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateSimilarity(Vector<>,Vector<>)` | Calculates similarity between two embeddings using cosine similarity. |
| `GetNeighbors(String)` | Gets all neighbors (both incoming and outgoing) of a node. |
| `Retrieve(Vector<>,Int32,Int32,Int32)` | Retrieves relevant nodes using hybrid vector + graph approach. |
| `RetrieveAsync(Vector<>,Int32,Int32,Int32)` | Retrieves relevant nodes asynchronously using hybrid approach. |
| `RetrieveWithRelationships(Vector<>,Int32,Dictionary<String,Double>,Int32)` | Retrieves nodes with relationship-aware scoring. |

