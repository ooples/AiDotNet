---
title: "VectorRetriever<T>"
description: "A dense vector-based retriever that uses embedding similarity for document retrieval."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Retrievers`

A dense vector-based retriever that uses embedding similarity for document retrieval.

## For Beginners

This retriever finds documents by meaning, not just keywords.

Think of it like a smart librarian who understands what you're asking:

- You ask: "How do cars work?"
- Keyword search finds: Documents with exact words "cars" and "work"
- Vector search finds: Documents about automobiles, engines, mechanics (similar meaning)

How it works:

1. Convert your question to a vector (list of numbers representing meaning)
2. Compare to vectors of all documents in the store
3. Find documents with closest vectors (most similar meaning)
4. Return the top matches

For example:

- Query: "renewable energy"
- Finds: Documents about solar, wind, hydroelectric (even if they don't say "renewable")
- Misses: Documents about fossil fuels (different meaning)

## How It Works

This retriever uses vector embeddings to find semantically similar documents. It embeds
the query using an embedding model, then searches the document store for the most similar
document vectors. This approach captures semantic meaning rather than just keyword matching.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VectorRetriever(IDocumentStore<>,IEmbeddingModel<>,Int32)` | Initializes a new instance of the VectorRetriever class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RetrieveCore(String,Int32,Dictionary<String,Object>)` | Core retrieval logic using dense vector similarity. |

