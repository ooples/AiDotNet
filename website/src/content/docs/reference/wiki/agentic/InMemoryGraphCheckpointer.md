---
title: "InMemoryGraphCheckpointer<TState>"
description: "An in-memory `IGraphCheckpointer` that keeps each thread's checkpoint history in a dictionary."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Graph.Checkpointing`

An in-memory `IGraphCheckpointer` that keeps each thread's checkpoint history in a
dictionary. Suitable for tests, single-process runs, and as the default when no durable store is wired.

## For Beginners

Keeps your save-games in memory. Great for tests and short-lived runs; for
durability across process restarts, use a database-backed checkpointer (coming in later slices).

## How It Works

Thread-safe via a per-instance lock. Because it stores the state object by reference, callers that use
a mutable reference type for `TState` should treat the state as immutable per step
(return a new/copied instance from nodes) to get true snapshots; durable checkpointers serialize and so
snapshot inherently.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAsync(String,String,CancellationToken)` |  |
| `GetHistoryAsync(String,CancellationToken)` |  |
| `GetLatestAsync(String,CancellationToken)` |  |
| `SaveAsync(GraphCheckpoint<>,CancellationToken)` |  |

