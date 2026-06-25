---
title: "StateGraph<TState>"
description: "A builder for a typed state graph: register nodes (state transformers), wire them with fixed or conditional edges (cycles allowed), set an entry point, then `Compile` into an executable `CompiledStateGraph`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Graph`

A builder for a typed state graph: register nodes (state transformers), wire them with fixed or
conditional edges (cycles allowed), set an entry point, then `Compile` into an executable
`CompiledStateGraph`.

## For Beginners

Think of building a flowchart. `Task{` adds a box that does
some work on your data; `String)` draws an arrow to the next box; `String})`
draws arrows that depend on the data; `String)` marks where to start. Arrows can
loop back to create cycles. `Compile` turns the flowchart into something you can run.

## How It Works

This is the AiDotNet counterpart to LangGraph's `StateGraph`, but fully typed: there are no
stringly-typed state dictionaries — `TState` is your own type, checked by the
compiler. Routing is explicit: a node has at most one fixed edge or one conditional router; reaching
`End` finishes the run.

**Example:**

## Methods

| Method | Summary |
|:-----|:--------|
| `AddConditionalEdges(String,Func<,String>)` | Adds conditional edges: after `from` runs, the router inspects the state and returns the next node name (or `End`). |
| `AddEdge(String,String)` | Adds a fixed edge: after `from` runs, flow always moves to `to`. |
| `AddFanOutNode(String,Func<,IEnumerable<>>,Func<,CancellationToken,Task<>>,Func<,IReadOnlyList<>,>,Nullable<Int32>)` | Adds a dynamic fan-out (map-reduce) node: it derives a set of items from the current state, runs a branch over each item in parallel, then reduces the branch results back into the state. |
| `AddInterruptBefore(String)` | Marks a node as a human-in-the-loop interrupt point: a run pauses just before this node so a human can review (and optionally edit) the state, then resume. |
| `AddNode(String,Func<,>)` | Adds a synchronous node that transforms the state. |
| `AddNode(String,Func<,CancellationToken,Task<>>)` | Adds an asynchronous node that transforms the state. |
| `AddRewardGatedEdges(String,Func<,Double>,Double,String,String)` | Adds reward-gated routing after a node: a scoring function rates the state, and flow goes to `ifMeetsThreshold` when the score is at least `threshold`, else to `ifBelowThreshold`. |
| `AddSubgraph(String,CompiledStateGraph<>)` | Adds a node that runs an entire compiled graph as a single step (a subgraph), threading the same state in and out. |
| `Compile` | Validates the graph and produces an executable `CompiledStateGraph`. |
| `SetEntryPoint(String)` | Sets the node where execution begins. |

## Fields

| Field | Summary |
|:-----|:--------|
| `End` | The terminal node name; route here to finish the run. |

