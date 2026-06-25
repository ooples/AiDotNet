---
title: "WALOperationType"
description: "Types of operations that can be logged in the WAL."
section: "API Reference"
---

`Enums` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Types of operations that can be logged in the WAL.

## Fields

| Field | Summary |
|:-----|:--------|
| `AddEdge` | Add an edge to the graph. |
| `AddNode` | Add a node to the graph. |
| `BeginTransaction` | Begin a transaction. |
| `Checkpoint` | Checkpoint - all operations up to this point are persisted. |
| `CommitTransaction` | Commit a transaction. |
| `RemoveEdge` | Remove an edge from the graph. |
| `RemoveNode` | Remove a node from the graph. |
| `RollbackTransaction` | Rollback a transaction. |

