---
title: "AiDotNetMeaiChatClient<T>"
description: "Adapts an AiDotNet `IChatClient` to a `IChatClient`, so AiDotNet models (including the in-process `LocalEngineChatClient` and the first-party connectors) can be consumed by any code written against the .NET ecosystem's standard chat abstrac…"
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Connectors`

Adapts an AiDotNet `IChatClient` to a `IChatClient`, so
AiDotNet models (including the in-process `LocalEngineChatClient` and the first-party connectors) can
be consumed by any code written against the .NET ecosystem's standard chat abstraction — Semantic Kernel,
the MEAI middleware pipeline, `FunctionInvokingChatClient`, etc.

## For Beginners

The other adapter lets AiDotNet use the ecosystem's models. This one is the
reverse: it makes *your* AiDotNet model look like a standard .NET chat model, so tools and apps that
only know the standard interface can use it without knowing it's AiDotNet underneath.

## How It Works

This is the outbound counterpart to `MeaiChatClient`. It maps MEAI messages/options/tools to
AiDotNet's types (via `MeaiInterop`), calls the wrapped client, and maps the response back —
including tool calls: an AiDotNet `ToolCallContent` surfaces as a MEAI
`FunctionCallContent`, so a MEAI host can drive AiDotNet's tool-calling loop.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AiDotNetMeaiChatClient(IChatClient<>)` | Initializes a new MEAI-facing adapter around an AiDotNet chat client. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `GetResponseAsync(IEnumerable<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetService(Type,Object)` |  |
| `GetStreamingResponseAsync(IEnumerable<ChatMessage>,ChatOptions,CancellationToken)` |  |

