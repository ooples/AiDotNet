---
title: "MemoryAugmentedAgent<T>"
description: "Wraps any `IAgent` with long-term memory recall: before each run it searches an `IAgentMemoryStore` for memories relevant to the latest user message and injects them as context, so the agent answers with knowledge gathered across previous c…"
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Memory`

Wraps any `IAgent` with long-term memory recall: before each run it searches an
`IAgentMemoryStore` for memories relevant to the latest user message and injects them as
context, so the agent answers with knowledge gathered across previous conversations.

## For Beginners

Before answering, the assistant flips through its long-term notes, pulls out
the ones related to your question, and reads them first — so it can use things it learned in earlier,
separate chats. It doesn't decide on its own what to write down; your app does that.

## How It Works

This is retrieval-augmented *memory* (RAG over the agent's own remembered facts). It composes with
the rest of the stack: the inner agent may be an `AgentExecutor`, a
`SupervisorAgent`, or a `Swarm`, and the result can in turn be wrapped by a
`ThreadedAgent` for short-term conversation memory. Writing memories is intentionally
explicit (call `CancellationToken)`) — this wrapper only reads, so what gets
remembered stays under the application's control.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemoryAugmentedAgent(IAgent<>,IAgentMemoryStore,MemoryAugmentationOptions)` | Initializes a new memory-augmented wrapper. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `RunAsync(IReadOnlyList<ChatMessage>,CancellationToken)` |  |

