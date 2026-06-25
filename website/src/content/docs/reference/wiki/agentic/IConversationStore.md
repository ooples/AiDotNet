---
title: "IConversationStore"
description: "Persists multi-turn conversations keyed by a thread id, so an agent can remember earlier turns across separate runs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Memory`

Persists multi-turn conversations keyed by a thread id, so an agent can remember earlier turns across
separate runs. This is the short-term ("thread") memory the agent stack composes via
`ThreadedAgent`.

## For Beginners

This is the notebook where a chat's history is written down under a label
(the thread id). Next time the same conversation continues, the agent reads the notebook first so it
remembers what was already said.

## How It Works

A store keeps an ordered list of dialogue messages per thread (typically user and assistant turns).
Implementations range from in-process (`InMemoryConversationStore`) to durable
(`JsonFileConversationStore`), with the same contract, so callers swap persistence without
changing agent code. Implementations persist a message's role and text; multimodal parts and tool-call
metadata are not part of the durable dialogue.

## Methods

| Method | Summary |
|:-----|:--------|
| `AppendAsync(String,IReadOnlyList<ChatMessage>,CancellationToken)` | Appends messages to the end of a thread's history, creating the thread if it does not exist. |
| `ClearAsync(String,CancellationToken)` | Removes a thread and its history. |
| `GetAsync(String,CancellationToken)` | Gets the ordered message history for a thread, or an empty list when the thread is unknown. |
| `ListThreadsAsync(CancellationToken)` | Lists the ids of all known threads. |

