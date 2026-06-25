---
title: "CompiledStateGraph<TState>"
description: "An executable, validated state graph produced by `Compile`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Graph`

An executable, validated state graph produced by `Compile`. Runs nodes
according to the graph's edges, threading the state from one node to the next until it reaches the end.

## For Beginners

This is the "compiled program" form of your graph. You give it a starting
state and it walks the nodes you wired up, handing each node the latest state, until it reaches the
end — then returns the final state.

## How It Works

Execution starts at the entry node. After a node runs, the next node is chosen by: its conditional
router (if any), else its fixed edge, else the run ends. Cycles are allowed and bounded by
`MaxSteps`. Use `CancellationToken)` for the final state or
`CancellationToken)` to observe each step.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetRecordedStateAsync(IGraphCheckpointer<>,String,CancellationToken)` | Returns the final recorded state for a thread (read from checkpoints, no execution). |
| `InvokeAsync(,GraphRunOptions,CancellationToken)` | Runs the graph from the supplied initial state and returns the final state. |
| `InvokeAsync(,IGraphCheckpointer<>,String,GraphRunOptions,CancellationToken)` | Runs the graph with durable checkpointing under a thread id, resuming automatically from the latest saved checkpoint if one exists for that thread (otherwise starting fresh from `initialState`). |
| `ReplayAsync(IGraphCheckpointer<>,String,CancellationToken)` | Deterministically replays a recorded run from its checkpoint history WITHOUT executing any nodes, yielding the same (node, state) sequence the original run produced. |
| `ResumeFromAsync(IGraphCheckpointer<>,String,String,GraphRunOptions,CancellationToken)` | Resumes (replays) a thread from a specific past checkpoint — the basis for time-travel — continuing execution from that checkpoint's next node and state, and saving new checkpoints as it goes. |
| `RunAsync(,IGraphCheckpointer<>,String,Func<,>,GraphRunOptions,CancellationToken)` | Runs the graph with human-in-the-loop interrupts under a thread id. |
| `StreamAsync(,GraphRunOptions,CancellationToken)` | Runs the graph and yields an update after each node executes, ending when flow reaches the end node. |

