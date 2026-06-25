---
title: "TracingAgent<T>"
description: "Wraps any `IAgent` and records each run as an `AgentTrajectory` in a `ITrajectoryStore`, without altering the agent's behavior."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.SelfImproving`

Wraps any `IAgent` and records each run as an `AgentTrajectory` in a
`ITrajectoryStore`, without altering the agent's behavior. This is how the self-improving
layer collects the experience it later evaluates and learns from.

## For Beginners

Put this around an agent and every time it runs, a copy of what happened is
filed away for later study. The agent itself behaves no differently; you just end up with a logbook.

## How It Works

Tracing is transparent and composable: the wrapper returns exactly what the inner agent returns and can
itself wrap (or be wrapped by) memory, supervisor, or swarm agents. Capture failures never affect the
run — the result is returned regardless of whether the store accepts the trajectory.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TracingAgent(IAgent<>,ITrajectoryStore,IReadOnlyDictionary<String,String>)` | Initializes a new tracing wrapper. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `RunAsync(IReadOnlyList<ChatMessage>,CancellationToken)` |  |

