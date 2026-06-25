---
title: "DelegateToolMiddleware"
description: "An `IToolMiddleware` backed by a delegate — the quick way to add a one-off tool filter (logging, argument fix-up, a guard) without declaring a class."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

An `IToolMiddleware` backed by a delegate — the quick way to add a one-off tool filter
(logging, argument fix-up, a guard) without declaring a class.

## For Beginners

The do-it-yourself tool filter: give it a function that receives the call and
the "run the tool" handle, and it becomes a reusable wrapper.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DelegateToolMiddleware(Func<ToolInvocationContext,ToolPipelineDelegate,CancellationToken,Task<ToolInvocationResult>>)` | Initializes a new delegate tool middleware. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeAsync(ToolInvocationContext,ToolPipelineDelegate,CancellationToken)` |  |

