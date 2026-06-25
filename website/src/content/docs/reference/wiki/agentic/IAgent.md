---
title: "IAgent<T>"
description: "A named, runnable agent: given a conversation, it produces a final answer (after optionally using tools and/or delegating to other agents along the way)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Agents`

A named, runnable agent: given a conversation, it produces a final answer (after optionally using
tools and/or delegating to other agents along the way).

## For Beginners

An agent is a worker with a `Name` and a one-line
`Description` of what it's good at. You hand it the conversation so far and it hands back a
result. Whether it solved the task alone or quietly asked teammates for help is its own business — callers
only see the final answer.

## How It Works

This is the single abstraction the multi-agent layer composes. A leaf agent (`AgentExecutor`)
drives a chat model in a tool-calling loop; a coordinator (supervisor/swarm) is itself an
`IAgent` that routes work to other `IAgent` members. Because the contract is
uniform, agents nest: a supervisor can supervise supervisors, and any agent can be exposed to another as a
callable tool.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets a short natural-language description of the agent's specialty, used by coordinators (and by models, when the agent is surfaced as a tool) to decide when to route work here. |
| `Name` | Gets the unique, stable name other agents and routers use to reference this agent. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RunAsync(IReadOnlyList<ChatMessage>,CancellationToken)` | Runs the agent over the supplied conversation and returns its result. |

