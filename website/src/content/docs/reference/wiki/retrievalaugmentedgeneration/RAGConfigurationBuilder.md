---
title: "RAGConfigurationBuilder<T>"
description: "Builder for constructing RAG configuration."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.RetrievalAugmentedGeneration.Configuration`

Builder for constructing RAG configuration.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RAGConfigurationBuilder` | Initializes a new instance of the `RAGConfigurationBuilder` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Build` | Builds the RAG configuration. |
| `WithChunking(String,Int32,Int32)` | Configures the chunking strategy. |
| `WithContextCompression(String,Double,Int32)` | Configures context compression. |
| `WithDocumentStore(String,Dictionary<String,Object>)` | Configures the document store. |
| `WithEmbedding(String,String,String,Int32)` | Configures the embedding model. |
| `WithQueryExpansion(String,Int32)` | Configures query expansion. |
| `WithReranking(String,Int32)` | Configures the reranking strategy. |
| `WithRetrieval(String,Int32)` | Configures the retrieval strategy. |

