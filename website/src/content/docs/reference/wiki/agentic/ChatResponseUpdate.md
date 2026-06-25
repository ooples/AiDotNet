---
title: "ChatResponseUpdate"
description: "A single incremental update in a streaming chat response."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models`

A single incremental update in a streaming chat response.

## For Beginners

This is one "frame" of the model typing in real time. Each frame usually
holds the next little bit of text (`TextDelta`). The last frame tells you it finished
and how many tokens were used. Use the factory methods (`String)`, `StreamingToolCallUpdate)`,
`ChatUsage)`) to build updates without juggling many constructor arguments.

## How It Works

Streaming delivers a response as a sequence of these updates instead of one final object. A typical
stream carries the role once, then many `TextDelta` chunks (and/or
`ToolCall` fragments), and finally a `FinishReason` with optional
`Usage`. Concatenating the deltas reconstructs the full reply.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChatResponseUpdate(Nullable<ChatRole>,String,StreamingToolCallUpdate,Nullable<ChatFinishReason>,ChatUsage)` | Initializes a new streaming update. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FinishReason` | Gets the finish reason, present only on the terminal update. |
| `Role` | Gets the author role for the response, when carried by this update. |
| `TextDelta` | Gets the next fragment of assistant text, or `null` when this update carries none. |
| `ToolCall` | Gets a streaming tool-call fragment, or `null` when this update carries none. |
| `Usage` | Gets token usage, present only on the terminal update for providers that report it mid-stream. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForFinish(ChatFinishReason,ChatUsage)` | Creates the terminal update carrying the finish reason and optional usage. |
| `ForText(String)` | Creates an update carrying a fragment of assistant text. |
| `ForToolCall(StreamingToolCallUpdate)` | Creates an update carrying a streaming tool-call fragment. |

