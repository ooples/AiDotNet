---
title: "AgentExecutor<T>"
description: "A single-model agent that drives an `IChatClient` in a native tool-calling loop: it calls the model, runs any tools the model requests, feeds the results back, and repeats until the model returns a final answer (or the iteration cap is hit)…"
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Agents`

A single-model agent that drives an `IChatClient` in a native tool-calling loop:
it calls the model, runs any tools the model requests, feeds the results back, and repeats until the
model returns a final answer (or the iteration cap is hit).

## For Beginners

Give it a model and (optionally) a toolbox. Ask it something. Internally it
loops: "model, here's the task and your tools" → if the model asks to use a tool, the executor runs it
and tells the model the result → repeat → until the model just answers. You get that final answer back,
plus the full transcript of what happened.

## How It Works

This replaces the legacy prompt-parsed ReAct loop with provider-native function calling: tool requests
arrive as structured `ToolCallContent` rather than scraped from prose, and tool results go
back as `Tool` messages. It is the leaf `IAgent` that the multi-agent
coordinators (supervisor/swarm) compose.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AgentExecutor(IChatClient<>,ToolCollection,AgentExecutorOptions)` | Initializes a new `AgentExecutor`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `RunAsync(IReadOnlyList<ChatMessage>,CancellationToken)` |  |
| `RunAsync(String,CancellationToken)` | Runs the agent against a single user message. |

