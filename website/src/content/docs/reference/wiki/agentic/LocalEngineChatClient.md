---
title: "LocalEngineChatClient<T>"
description: "An `IChatClient` that runs entirely in-process over an `ICausalLanguageModel` — no network, no API key, no external service."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models.Local`

An `IChatClient` that runs entirely in-process over an `ICausalLanguageModel`
— no network, no API key, no external service. It renders the conversation to a prompt, encodes it,
autoregressively samples tokens until the end-of-sequence token or a length limit, and decodes the
result. This is the flagship "local-first" capability: the same agent code that drives OpenAI or
Anthropic drives AiDotNet's own model.

## For Beginners

This is your own chatbot brain running on your machine. You hand it the
conversation; it writes the reply one word-piece at a time until it decides it's done or hits the length
cap. Everything else in this library that talks to a "chat model" can talk to this one instead — so you
can build agents with no cloud dependency at all.

## How It Works

Because it implements `IChatClient`, the local engine is a drop-in for every higher layer
(agents, supervisor/swarm, memory). Both non-streaming and streaming generation are supported; streaming
decodes incrementally and yields the new text on each step. Constrained decoding *is* supported via
`Constraint` (an `ITokenConstraint` enforced at the logits, e.g.
`AllowedTokenSetConstraint` / `FiniteStateTokenConstraint`). What this slice does
not do: native tool-calling and auto-deriving a constraint from `ResponseFormat` —
requests that ask for either are rejected with `NotSupportedException` rather than silently
returning plain text; set `Constraint` explicitly for guaranteed-structured
output.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LocalEngineChatClient(ICausalLanguageModel<>,IGenerationTokenizer,IChatPromptTemplate,LocalEngineOptions)` | Initializes a new local engine. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelId` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetStreamingResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |

