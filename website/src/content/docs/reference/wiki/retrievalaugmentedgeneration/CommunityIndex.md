---
title: "CommunityIndex<T>"
description: "Index structure that maps hierarchy levels and community IDs to their summaries, enabling efficient community-based retrieval for GraphRAG."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Communities`

Index structure that maps hierarchy levels and community IDs to their summaries,
enabling efficient community-based retrieval for GraphRAG.

## For Beginners

The community index is like a table of contents for your knowledge graph.
Instead of searching every node, you can search community summaries first to quickly find
which part of the graph is relevant to your query.

## Methods

| Method | Summary |
|:-----|:--------|
| `Build(KnowledgeGraph<>,LeidenResult,Int32,Int32)` | Builds the index from a knowledge graph and Leiden result. |
| `GetSummariesAtLevel(Int32)` | Gets all community summaries at a given level. |
| `GetSummary(Int32,Int32)` | Gets a community summary by level and community ID. |
| `SearchCommunities(String,Int32,Int32)` | Searches community summaries for those relevant to a query string. |

