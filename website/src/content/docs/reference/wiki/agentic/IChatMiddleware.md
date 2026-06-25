---
title: "IChatMiddleware"
description: "A cross-cutting filter around chat-model calls (the AiDotNet analogue of Semantic Kernel filters / a middleware pipeline)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Pipeline`

A cross-cutting filter around chat-model calls (the AiDotNet analogue of Semantic Kernel filters / a
middleware pipeline). Each middleware wraps the next stage: it can modify the request before calling
`next`, inspect or replace the response after, short-circuit by returning without calling
next (caching, guardrails, mocking), retry, log, or measure.

## For Beginners

A reusable wrapper you can put around *every* model call: log it, time
it, add a standing instruction, block unsafe content, cache repeats, or retry on failure — without
touching the agent or the model. Stack several and they run in order.

## How It Works

Middleware are composed by `MiddlewareChatClient` in registration order (the first
registered runs outermost). Because the request/response types are not numeric-generic, middleware are
written once and apply to any `IChatClient` backend — cloud or local.

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeAsync(ChatRequestContext,ChatPipelineDelegate,CancellationToken)` | Processes a chat call, optionally calling `next` to continue down the pipeline. |

