---
title: "ReplayingChatClient<T>"
description: "An `IChatClient` that serves responses from a recorded `IChatInteractionStore` — deterministic replay without calling any model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

An `IChatClient` that serves responses from a recorded `IChatInteractionStore` —
deterministic replay without calling any model. On a cache miss it falls through to an optional inner
client (recording the new interaction) or throws, depending on configuration.

## For Beginners

Plays back saved model answers. Ask the same thing and you get the same recorded
reply instantly — no model call. Great for fast, deterministic tests and for re-running a session exactly.

## How It Works

This makes agent runs reproducible: record once against a real model, then replay the exact same
trajectory in tests/CI/debugging at zero cost and with no nondeterminism. With a fallback client it acts
as a persistent cache (replay hits are free; misses call the model and are recorded).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReplayingChatClient(IChatInteractionStore,IChatClient<>,String)` | Initializes a new replaying client. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelId` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetStreamingResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |

