---
title: "EnhancedGraphRAG<T>"
description: "Enhanced Graph-based RAG that integrates with `KnowledgeGraph` and supports Local, Global (Leiden community summaries), and DRIFT retrieval modes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Enhanced Graph-based RAG that integrates with `KnowledgeGraph` and supports
Local, Global (Leiden community summaries), and DRIFT retrieval modes.

## For Beginners

EnhancedGraphRAG combines three retrieval strategies:

**Local Search:** Best for specific factual questions.

1. Find entities in the graph matching the query
2. Traverse their neighborhood (1-2 hops)
3. Collect related entities and their connections as context

**Global Search:** Best for broad thematic questions ("What are all the research areas?").

1. Pre-compute community summaries via Leiden algorithm
2. Search community summaries for relevant communities
3. Return community descriptions as context

**DRIFT Search:** Best for complex queries needing both breadth and depth.

1. Start with Global search to identify relevant communities
2. From top communities, pick key entities
3. Run Local search from those entities to refine context
4. Repeat refinement for N iterations

## How It Works

Unlike the existing `GraphRAG`,
this class works directly with the `KnowledgeGraph` class (which delegates to IGraphStore),
rather than maintaining its own internal dictionary. It also adds Leiden-based community summarization
for global and DRIFT search modes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnhancedGraphRAG(KnowledgeGraph<>,GraphRAGOptions)` | Creates a new EnhancedGraphRAG instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CommunityStructure` | Gets the Leiden community detection result, if community detection has been run. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildCommunityIndex` | Builds the community index for Global and DRIFT search modes. |
| `DriftSearch(String,Int32)` | DRIFT-inspired search: Dynamic Reasoning and Inference with Flexible Traversal. |
| `Retrieve(String,Int32)` | Retrieves context from the knowledge graph for a given query. |
| `RetrieveNodes(String,Int32)` | Retrieves relevant graph nodes for a given query. |

