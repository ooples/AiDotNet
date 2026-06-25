---
title: "MiddlewareChatClient<T>"
description: "An `IChatClient` decorator that runs a chain of `IChatMiddleware` around an inner client's calls — the composition root for filters/middleware (logging, guardrails, caching, retry, telemetry)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

An `IChatClient` decorator that runs a chain of `IChatMiddleware` around an
inner client's calls — the composition root for filters/middleware (logging, guardrails, caching, retry,
telemetry). The first-registered middleware runs outermost.

## For Beginners

Wrap your model client in this with a list of filters, and every
(non-streaming) call flows through them in order before and after hitting the model — one place to add
behavior that should apply everywhere.

## How It Works

Middleware semantics (pre/post processing, short-circuiting) are defined on the complete-response path.
When middleware are configured, `CancellationToken)` therefore runs the full pipeline
and re-emits the final response as a stream — streaming callers get the exact same policy enforcement as
non-streaming callers, at the cost of token-level incrementality. With no middleware configured, streaming
passes through to the inner client natively.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MiddlewareChatClient(IChatClient<>,IReadOnlyList<IChatMiddleware>)` | Initializes a new middleware client. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelId` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |
| `GetStreamingResponseAsync(IReadOnlyList<ChatMessage>,ChatOptions,CancellationToken)` |  |

