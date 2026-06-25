---
title: "InMemoryConversationStore"
description: "A process-local `IConversationStore` that keeps thread histories in memory."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Memory`

A process-local `IConversationStore` that keeps thread histories in memory. Ideal for tests,
single-process apps, and the default zero-config experience; histories are lost when the process exits.

## For Beginners

The simplest notebook — kept in RAM. Fast and needs no setup, but forgotten
when the program stops. For history that survives restarts, use `JsonFileConversationStore`
(or a database-backed store).

## Methods

| Method | Summary |
|:-----|:--------|
| `AppendAsync(String,IReadOnlyList<ChatMessage>,CancellationToken)` |  |
| `ClearAsync(String,CancellationToken)` |  |
| `GetAsync(String,CancellationToken)` |  |
| `ListThreadsAsync(CancellationToken)` |  |

