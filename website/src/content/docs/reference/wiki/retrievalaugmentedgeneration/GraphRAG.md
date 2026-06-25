---
title: "GraphRAG<T>"
description: "Graph-based RAG (Retrieval Augmented Generation) that combines knowledge graph traversal with vector search for enhanced retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns`

Graph-based RAG (Retrieval Augmented Generation) that combines knowledge graph traversal with vector search for enhanced retrieval.

## For Beginners

Think of a knowledge graph like a mind map or family tree of facts.

Traditional RAG (vector-only):

- Question: "What did Einstein discover?"
- Search embeddings for documents about "Einstein" and "discovery"
- Problem: Might miss important connections or relationships

GraphRAG (graph + vector):

- Question: "What did Einstein discover?"
- Step 1: Extract entities → "Einstein"
- Step 2: Check knowledge graph:
* Einstein → DISCOVERED → Theory of Relativity
* Einstein → WORKED_AT → Princeton University
* Theory of Relativity → INFLUENCED → Quantum Mechanics
- Step 3: Vector search for "Einstein" and "discovery"
- Step 4: Boost documents that mention graph-connected entities
- Result: Prioritizes documents about his actual discoveries over generic biographical info

Real-world analogy:

- Regular search: Looking through books by reading every page
- GraphRAG: Using the index AND table of contents AND cross-references all together

## How It Works

GraphRAG enhances traditional RAG by maintaining a knowledge graph of entities and relationships alongside
vector embeddings. When a query mentions entities in the graph, GraphRAG retrieves both directly related
documents (via graph traversal) and semantically similar documents (via vector search), then boosts scores
for documents that appear in both results. This leverages structured knowledge for more accurate retrieval.

**Example Usage:**

**How It Works:**
The retrieval process:

1. Entity Extraction - Use LLM to extract entities from the query
2. Graph Traversal - Find all entities connected to query entities in the knowledge graph
3. Vector Retrieval - Perform standard semantic search for the query
4. Score Boosting - Multiply scores by 1.5x for documents mentioning graph-connected entities
5. Ranking - Sort all documents by boosted scores and return top-K

Current implementation uses:

- In-memory dictionary for knowledge graph (production should use Neo4j, GraphDB)
- Regex-based entity extraction (production should use NER models)
- Simple score boosting (production could use graph embeddings, PageRank, etc.)

**Benefits:**

- Structured reasoning - Leverages explicit relationships between entities
- Better precision - Prioritizes documents with known connections
- Explainable - Can trace why documents were selected via graph paths
- Handles multi-hop reasoning - Can traverse entity → relation → entity chains
- Complementary - Combines structured (graph) and unstructured (vector) knowledge

**Limitations:**

- Requires building/maintaining knowledge graph (initial overhead)
- Graph quality affects results (garbage in, garbage out)
- Entity extraction quality matters (missed entities = missed connections)
- Current implementation is in-memory only (not scalable for large graphs)
- Simple boosting strategy (more sophisticated approaches possible)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphRAG(IGenerator<>,RetrieverBase<>,Int32)` | Initializes a new instance of the `GraphRAG` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddRelation(String,String,String)` | Adds a relationship to the knowledge graph. |
| `Retrieve(String,Int32)` | Retrieves documents using combined knowledge graph traversal and vector similarity search. |

