---
title: "CohereChatClient<T>"
description: "An `IChatClient` for Cohere's Chat API, whose bespoke wire format splits a turn into the latest `message`, a `chat_history` of prior turns, and a `preamble` for system instructions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Connectors`

An `IChatClient` for Cohere's Chat API, whose bespoke wire format splits a turn into the
latest `message`, a `chat_history` of prior turns, and a `preamble` for system instructions.
This connector maps the unified agentic types to and from that shape, including tool declarations and
tool-call parsing.

## For Beginners

Same agent code, pointed at Cohere. The class rearranges the conversation into
the layout Cohere expects (most recent message separate from the history) and translates the reply back.

## How It Works

Streaming is a single-shot fallback (complete answer as one update). Tool calling round-trips in Cohere's
native format: assistant tool-call turns carry `tool_calls` in the history, and tool results are sent
as `tool_results` (promoted to the live request leg when they are the latest turn).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CohereChatClient(String,String,String,HttpClient)` | Initializes a new Cohere client. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetResponseCoreAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetStreamingResponseCoreAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |

