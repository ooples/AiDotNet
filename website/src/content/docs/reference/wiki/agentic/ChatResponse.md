---
title: "ChatResponse"
description: "The complete result of a non-streaming chat call."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models`

The complete result of a non-streaming chat call.

## For Beginners

This is everything you get back from one call: the reply itself, the
reason it stopped, and how many tokens it cost. If `FinishReason` is
`ToolCalls`, the reply is asking you to run tools rather than giving a
final answer.

## How It Works

Wraps the assistant `Message` the model produced (which may contain text and/or
`ToolCallContent` parts), why generation stopped (`FinishReason`), token
`Usage`, and the model id that served the request.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChatResponse(ChatMessage,ChatFinishReason,ChatUsage,String)` | Initializes a new `ChatResponse`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FinishReason` | Gets the reason generation stopped. |
| `Message` | Gets the assistant message produced by the model. |
| `ModelId` | Gets the id of the model that served the request, or `null` when not reported. |
| `Text` | Gets the concatenated text of the assistant message (shortcut for `Message.Text`). |
| `Usage` | Gets token usage for the request, or `null` when the provider did not report it. |

