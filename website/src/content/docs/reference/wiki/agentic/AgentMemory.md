---
title: "AgentMemory"
description: "A single long-term memory: a piece of text the agent should be able to recall later, with a stable id and optional metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Memory`

A single long-term memory: a piece of text the agent should be able to recall later, with a stable id
and optional metadata. Memories live across conversation threads (unlike `IConversationStore`,
which is per-thread short-term history).

## For Beginners

Think of one sticky note the assistant keeps — e.g. "the user prefers metric
units" or "the project deadline is in June". Each note has a unique id (so it can be updated or removed)
and the note text itself. Later, when something relevant comes up, the assistant can find and re-read the
note even if it's in a completely different conversation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AgentMemory(String,String,IReadOnlyDictionary<String,String>)` | Initializes a new memory. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Content` | Gets the remembered text. |
| `Id` | Gets the stable, unique identifier for this memory. |
| `Metadata` | Gets optional key/value metadata, or `null` when none was supplied. |

