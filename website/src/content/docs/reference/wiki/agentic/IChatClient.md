---
title: "IChatClient<T>"
description: "A message-based chat model client that supports native tool calling, streaming, and structured output."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Models`

A message-based chat model client that supports native tool calling, streaming, and structured output.
This is the foundation the agentic orchestration subsystem is built on, superseding the legacy
text-in/text-out `IChatModel<T>` / `ILanguageModel<T>`.

## For Beginners

This is the "brain" interface every agent talks to. You give it the
conversation so far and some settings; it gives you back the next reply — either all at once
(`CancellationToken)`) or piece by piece as it's generated (`CancellationToken)`).
Because the interface is the same for every provider, you can swap a cloud model for a local one
without rewriting your agent.

## How It Works

Implementations translate a conversation (a list of `ChatMessage`) plus per-call
`ChatOptions` into a provider call, and return either a complete `ChatResponse`
or a stream of `ChatResponseUpdate` chunks. The same abstraction covers cloud providers
(OpenAI, Anthropic, Azure) and the local engine, so higher layers (tools, graph, agents) are written
once and run against any backend.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelId` | Gets the identifier of the underlying model (e.g., `gpt-4o`, `claude-3-5-sonnet`, or a local model name). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` | Sends the conversation and returns the model's complete response. |
| `GetStreamingResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` | Sends the conversation and streams the model's response as a sequence of incremental updates. |

