---
title: "JsonFileConversationStore"
description: "An `IConversationStore` that persists thread histories to a single JSON file, so conversations survive process restarts without an external database."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Memory`

An `IConversationStore` that persists thread histories to a single JSON file, so
conversations survive process restarts without an external database. Each message is stored as its role
plus its text (the durable dialogue); multimodal and tool-call parts are not persisted.

## For Beginners

The same notebook as the in-memory store, but written to a file on disk, so
closing and reopening your app keeps the conversation history.

## How It Works

All threads live in one file as a `{ threadId: [ {role, text}, ... ] }` map. Reads and writes are
serialized with an in-process lock and the whole file is rewritten on each append, which suits modest
single-process workloads. For concurrent or high-volume scenarios, use a database-backed store.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `JsonFileConversationStore(String)` | Initializes a store backed by the given file. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AppendAsync(String,IReadOnlyList<ChatMessage>,CancellationToken)` |  |
| `ClearAsync(String,CancellationToken)` |  |
| `GetAsync(String,CancellationToken)` |  |
| `ListThreadsAsync(CancellationToken)` |  |

