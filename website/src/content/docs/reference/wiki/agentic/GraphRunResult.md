---
title: "GraphRunResult<TState>"
description: "The outcome of a human-in-the-loop graph run: either the run completed, or it paused before an interrupt node awaiting input."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Graph`

The outcome of a human-in-the-loop graph run: either the run completed, or it paused before an
interrupt node awaiting input. Carries the state either way.

## For Beginners

Tells you whether the graph finished or stopped to ask a human. If it
stopped, `InterruptedBefore` says which step it's waiting to run, and `State`
is the data so far so you can review/edit it before continuing.

## How It Works

When a graph has interrupt points (see `String)`), a run
stops just before such a node and returns an interrupted result with `InterruptedBefore`
set to that node. Call the run again on the same thread to resume past the pause (optionally editing
the state with the human's input first).

## Properties

| Property | Summary |
|:-----|:--------|
| `InterruptedBefore` | Gets the node the run paused before, or `null` when the run completed. |
| `IsComplete` | Gets a value indicating whether the run reached the end node. |
| `IsInterrupted` | Gets a value indicating whether the run paused at an interrupt point. |
| `State` | Gets the state (final when complete, or as of the pause when interrupted). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Complete()` | Creates a completed result. |
| `Interrupted(,String)` | Creates an interrupted result. |

