---
title: "GraphCheckpoint<TState>"
description: "An immutable snapshot of a graph run at a point in time: which node runs next and the state as of that point."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Graph.Checkpointing`

An immutable snapshot of a graph run at a point in time: which node runs next and the state as of
that point. Saved after each step so a run can be resumed, inspected, or replayed.

## For Beginners

Like a save-game. It remembers exactly where the graph was about to go
next and what the data looked like, so you can stop and pick up later (or rewind to an earlier save).

## How It Works

A checkpoint records the `NextNode` (where execution will continue) plus the
`State` at that boundary, tagged with a `ThreadId` (the run/conversation) and a
monotonically increasing `Step`. The sequence of checkpoints for a thread is its history —
the basis for durable resume and time-travel.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphCheckpoint(String,String,Int32,String,)` | Initializes a new checkpoint. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointId` | Gets the unique id of this checkpoint within the thread. |
| `IsComplete` | Gets a value indicating whether this checkpoint represents a completed run. |
| `NextNode` | Gets the node that will execute next (or `End` when the run is complete). |
| `State` | Gets the state captured at this boundary. |
| `Step` | Gets the zero-based, monotonically increasing step index. |
| `ThreadId` | Gets the run/thread this checkpoint belongs to. |

