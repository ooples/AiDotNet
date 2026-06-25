---
title: "IGraphCheckpointer<TState>"
description: "Persists and retrieves `GraphCheckpoint`s, enabling durable resume and time-travel for graph runs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Graph.Checkpointing`

Persists and retrieves `GraphCheckpoint`s, enabling durable resume and time-travel
for graph runs. Implementations back onto memory, SQLite, Postgres, Redis, etc.

## For Beginners

The save-game store. The graph asks it to save a checkpoint after each
step, to fetch the latest checkpoint when resuming, and to list the full history when rewinding.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAsync(String,String,CancellationToken)` | Gets a specific checkpoint by id (used for time-travel / replay from a past point). |
| `GetHistoryAsync(String,CancellationToken)` | Gets the full `GraphCheckpoint` history for `threadId` in append (chronological) order — i.e. |
| `GetLatestAsync(String,CancellationToken)` | Gets the most recent checkpoint for a thread, or `null` if the thread has none. |
| `SaveAsync(GraphCheckpoint<>,CancellationToken)` | Saves (appends) a checkpoint. |

