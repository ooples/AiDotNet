---
title: "GeminiChatClient<T>"
description: "An `IChatClient` for Google's Gemini models via the `generateContent` API."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Connectors`

An `IChatClient` for Google's Gemini models via the `generateContent` API. Unlike the
OpenAI-compatible providers, Gemini has a bespoke wire format (contents/parts, systemInstruction,
generationConfig, functionDeclarations), which this connector maps to and from the unified agentic model
types — including native function calling and usage.

## For Beginners

Same agent code, pointed at Google Gemini. This class quietly translates
between AiDotNet's message format and Gemini's so everything else just works.

## How It Works

Streaming is provided as a single-shot fallback (it returns the complete answer as one update rather than
incremental tokens; tool-call deltas are not streamed) — use `CancellationToken)`
for native tool calling. Tool calling round-trips in Gemini's native format: assistant tool-call turns are
sent back as `functionCall` parts and tool results as `functionResponse` parts.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GeminiChatClient(String,String,String,HttpClient)` | Initializes a new Gemini client. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetResponseCoreAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetStreamingResponseCoreAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |

