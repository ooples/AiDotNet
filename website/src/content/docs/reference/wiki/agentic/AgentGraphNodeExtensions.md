---
title: "AgentGraphNodeExtensions"
description: "Bridges the agents layer and the graph runtime: adds an `IAgent` as a node in a `StateGraph`."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Agents`

Bridges the agents layer and the graph runtime: adds an `IAgent` as a node in a
`StateGraph`. The node maps the graph state to the agent's input, runs the agent, and
folds the result back into the state — so agents (executors, supervisors, swarms) become first-class steps
in a typed, checkpointable, resumable graph.

## For Beginners

The graph is a flowchart with state flowing through it; these helpers let one
box in the flowchart *be* an agent. You say how to read the agent's question out of the state and
how to write its answer back, and the graph handles the rest (routing, retries, saving progress).

## How It Works

This realizes the unified "typed deterministic graph + multi-agent" story: a graph can route between
agent nodes with conditional edges, cycles, human-in-the-loop interrupts, and durable checkpointing, while
each node is a full agent (which may itself coordinate a team).

## Methods

| Method | Summary |
|:-----|:--------|
| `AddAgentNode(StateGraph<>,String,IAgent<>,Func<,IReadOnlyList<ChatMessage>>,Func<,AgentRunResult,>)` | Adds a graph node that runs an agent against a full message list derived from the state (for richer control than a single user message). |
| `AddAgentNode(StateGraph<>,String,IAgent<>,Func<,String>,Func<,AgentRunResult,>)` | Adds a graph node that runs an agent against a single user message derived from the state. |

