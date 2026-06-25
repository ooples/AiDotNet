---
title: "EmbeddingAgentMemoryStore<T>"
description: "A semantic `IAgentMemoryStore` that ranks memories by meaning, not words."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Memory`

A semantic `IAgentMemoryStore` that ranks memories by meaning, not words. It embeds each
memory with an `IEmbeddingModel` and scores a query by cosine similarity against those
embeddings — reusing AiDotNet's RAG embedding and similarity-metric stack, so a memory about "due date"
is recalled for a query about "deadline".

## For Beginners

Same notebook idea, but instead of matching words it matches *meaning*.
It turns every note (and your question) into a list of numbers that capture meaning, then finds the notes
whose numbers are closest to your question's — so synonyms and paraphrases still match.

## How It Works

Memories and their embedding vectors are held in memory; the vector for each memory is computed once on
`CancellationToken)`. This keeps the abstraction identical to the lexical
`InMemoryAgentMemoryStore` while delivering true semantic recall — pick whichever fits the
deployment without touching agent code.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EmbeddingAgentMemoryStore(IEmbeddingModel<>,ISimilarityMetric<>)` | Initializes a new semantic memory store. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAsync(String,IReadOnlyDictionary<String,String>,CancellationToken)` |  |
| `GetAllAsync(CancellationToken)` |  |
| `RemoveAsync(String,CancellationToken)` |  |
| `SearchAsync(String,Int32,CancellationToken)` |  |

