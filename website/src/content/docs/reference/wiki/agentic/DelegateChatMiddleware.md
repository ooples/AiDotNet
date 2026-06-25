---
title: "DelegateChatMiddleware"
description: "An `IChatMiddleware` backed by a delegate — the quick way to add a one-off filter (logging, a header injection, a guard) without declaring a class."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

An `IChatMiddleware` backed by a delegate — the quick way to add a one-off filter (logging,
a header injection, a guard) without declaring a class.

## For Beginners

The do-it-yourself middleware: hand it a small function that receives the
request and the "call the rest of the pipeline" handle, and it becomes a reusable filter.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DelegateChatMiddleware(Func<ChatRequestContext,ChatPipelineDelegate,CancellationToken,Task<ChatResponse>>)` | Initializes a new delegate middleware. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeAsync(ChatRequestContext,ChatPipelineDelegate,CancellationToken)` |  |

