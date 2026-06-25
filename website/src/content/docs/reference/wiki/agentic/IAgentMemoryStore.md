---
title: "IAgentMemoryStore"
description: "A long-term, cross-thread memory store: it remembers facts and can retrieve the ones most relevant to a query."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Memory`

A long-term, cross-thread memory store: it remembers facts and can retrieve the ones most relevant to a
query. This is the durable counterpart to the per-thread `IConversationStore` and the
retrieval source behind `MemoryAugmentedAgent`.

## For Beginners

This is the assistant's notebook of long-term facts. You can add notes,
search for the notes most related to a question, list them, or remove one. Whether the search matches by
shared words or by meaning depends on which implementation you plug in — the way you use it is the same.

## How It Works

Implementations differ only in *how* they rank relevance: `InMemoryAgentMemoryStore`
uses lexical overlap (zero-config, no model), while `EmbeddingAgentMemoryStore<T>` uses an
`IEmbeddingModel` for true semantic similarity by reusing AiDotNet's
RAG embedding + cosine-metric stack. Callers depend only on this interface, so semantic search is a
drop-in upgrade over the lexical default.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAsync(String,IReadOnlyDictionary<String,String>,CancellationToken)` | Stores a new memory and returns its generated id. |
| `GetAllAsync(CancellationToken)` | Returns all stored memories (order unspecified). |
| `RemoveAsync(String,CancellationToken)` | Removes a memory by id. |
| `SearchAsync(String,Int32,CancellationToken)` | Returns the memories most relevant to a query, highest score first. |

