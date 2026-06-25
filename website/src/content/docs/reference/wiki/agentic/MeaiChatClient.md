---
title: "MeaiChatClient<T>"
description: "Adapts a `IChatClient` (the .NET ecosystem's standard chat abstraction) to AiDotNet's `IChatClient`, so any Microsoft.Extensions.AI connector (OpenAI, Azure, Ollama, etc.) can drive AiDotNet agents and reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Connectors`

Adapts a `IChatClient` (the .NET ecosystem's standard chat
abstraction) to AiDotNet's `IChatClient`, so any Microsoft.Extensions.AI connector
(OpenAI, Azure, Ollama, etc.) can drive AiDotNet agents and reasoning.

## For Beginners

Microsoft.Extensions.AI is .NET's shared interface for talking to chat
models. This adapter lets you take any model that speaks that interface and use it anywhere AiDotNet
expects its own `IChatClient` — so you inherit the whole ecosystem of providers.

## How It Works

This bridges the two ecosystems for text, sampling, streaming, *and tool calling*. AiDotNet tool
definitions are passed to the MEAI model as schema-only function declarations (via `MeaiInterop`);
tool calls the model requests come back as AiDotNet `ToolCallContent` with the finish reason set
to `ToolCalls`, and the AiDotNet agent loop executes them — MEAI never invokes
the tool itself. Tool results are replayed back to the model as MEAI function-result content.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MeaiChatClient(IChatClient,String)` | Initializes a new adapter around a Microsoft.Extensions.AI chat client. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelId` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetStreamingResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |

