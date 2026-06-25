---
title: "InMemoryAgentMemoryStore"
description: "A process-local `IAgentMemoryStore` that ranks memories by lexical overlap with the query — the fraction of the query's words that appear in the memory."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Memory`

A process-local `IAgentMemoryStore` that ranks memories by lexical overlap with the query —
the fraction of the query's words that appear in the memory. Requires no embedding model, so it is the
zero-config default; for meaning-based recall, use `EmbeddingAgentMemoryStore<T>`.

## For Beginners

This notebook finds notes by matching words. If you ask about "deadline" it
finds notes containing "deadline", but it won't realize "due date" means the same thing — that needs the
embedding-backed store. It's fast, needs no setup, and is great for tests and simple cases.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAsync(String,IReadOnlyDictionary<String,String>,CancellationToken)` |  |
| `GetAllAsync(CancellationToken)` |  |
| `RemoveAsync(String,CancellationToken)` |  |
| `SearchAsync(String,Int32,CancellationToken)` |  |

