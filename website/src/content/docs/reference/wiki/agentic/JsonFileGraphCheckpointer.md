---
title: "JsonFileGraphCheckpointer<TState>"
description: "A durable `IGraphCheckpointer` that persists all threads' checkpoints to a single JSON file, so runs survive process restarts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Graph.Checkpointing`

A durable `IGraphCheckpointer` that persists all threads' checkpoints to a single
JSON file, so runs survive process restarts. Zero extra dependencies (uses Newtonsoft.Json).

## For Beginners

Saves your graph's progress to a file on disk, so if the app restarts you
can resume right where it left off.

## How It Works

Simple and self-contained: every save reads, mutates, and rewrites the file under an in-process lock,
so it is correct for single-process durability but not tuned for high-throughput or multi-process
concurrent writers. For those, use a database-backed checkpointer (e.g., SQLite in
`AiDotNet.Storage.Sqlite`).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `JsonFileGraphCheckpointer(String)` | Initializes the checkpointer, creating the containing directory if needed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAsync(String,String,CancellationToken)` |  |
| `GetHistoryAsync(String,CancellationToken)` |  |
| `GetLatestAsync(String,CancellationToken)` |  |
| `SaveAsync(GraphCheckpoint<>,CancellationToken)` |  |

