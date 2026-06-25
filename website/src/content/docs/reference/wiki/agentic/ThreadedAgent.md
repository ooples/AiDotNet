---
title: "ThreadedAgent<T>"
description: "Wraps any `IAgent` with conversation memory: each run is tied to a thread id, so prior turns are loaded from an `IConversationStore` and prepended to the new input, and the new user/assistant turn is persisted afterwards."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Memory`

Wraps any `IAgent` with conversation memory: each run is tied to a thread id, so prior
turns are loaded from an `IConversationStore` and prepended to the new input, and the new
user/assistant turn is persisted afterwards. This is how a stateless agent becomes a stateful chat.

## For Beginners

A plain agent forgets everything between questions. Wrap it in a
`ThreadedAgent` with a thread id (like a chat-session id) and it remembers: it reads the
past conversation, answers with that context, and writes the new exchange back for next time.

## How It Works

The thread persists a clean user/assistant dialogue: each call appends the user message and the agent's
final answer. Intermediate tool calls and the system prompt remain internal to the inner agent's run and
are available on the returned `AgentRunResult` for that turn, but are not stored as part of
the durable conversation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ThreadedAgent(IAgent<>,IConversationStore)` | Initializes a new threaded wrapper. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the wrapped agent's description. |
| `Name` | Gets the wrapped agent's name. |

## Methods

| Method | Summary |
|:-----|:--------|
| `RunAsync(String,IReadOnlyList<ChatMessage>,CancellationToken)` | Continues a thread with one or more new messages. |
| `RunAsync(String,String,CancellationToken)` | Continues a thread with a single user message: loads the thread's history, runs the inner agent with that history plus the new message, then appends the user message and the agent's answer to the thread. |

