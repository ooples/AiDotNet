---
title: "AnthropicChatClient<T>"
description: "An `IChatClient` for Anthropic's Claude Messages API with native tool use, streaming, and multimodal (image) input."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Connectors`

An `IChatClient` for Anthropic's Claude Messages API with native tool use, streaming,
and multimodal (image) input.

## For Beginners

This adapter lets the library talk to Claude models. It hides Anthropic's
specific request/response format so the rest of the code uses the same messages and options as for any
other provider.

## How It Works

Anthropic's wire format differs from OpenAI's: system instructions are a top-level field (not a
message), message content is an array of typed blocks (`text`, `image`, `tool_use`,
`tool_result`), tool results are carried on user-role messages, and `max_tokens` is required.
This connector maps the provider-neutral model onto that shape and back, translating Claude's
`stop_reason` onto `ChatFinishReason`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnthropicChatClient(String,String,String,String,HttpClient)` | Initializes a new Anthropic chat client. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetResponseCoreAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetStreamingResponseCoreAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |

