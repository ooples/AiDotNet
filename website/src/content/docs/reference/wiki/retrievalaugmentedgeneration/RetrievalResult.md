---
title: "RetrievalResult<T>"
description: "Represents a retrieval result from the hybrid retriever."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Represents a retrieval result from the hybrid retriever.

## Properties

| Property | Summary |
|:-----|:--------|
| `Depth` | Gets or sets the graph traversal depth (0 for initial candidates). |
| `Embedding` | Gets or sets the embedding vector. |
| `NodeId` | Gets or sets the node ID. |
| `ParentNodeId` | Gets or sets the parent node ID (for graph-traversed results). |
| `RelationType` | Gets or sets the relationship type (for graph-traversed results). |
| `Score` | Gets or sets the relevance score (0-1, higher is more relevant). |
| `Source` | Gets or sets how this result was retrieved. |

