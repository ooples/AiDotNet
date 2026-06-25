---
title: "IToolMiddleware"
description: "A cross-cutting filter around tool/function execution (the AiDotNet analogue of Semantic Kernel's function-invocation filter)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Pipeline`

A cross-cutting filter around tool/function execution (the AiDotNet analogue of Semantic Kernel's
function-invocation filter). Middleware can rewrite arguments, require approval, log, time, cache, or
short-circuit a tool call (e.g., deny it) by returning a result without calling `next`.

## For Beginners

A reusable wrapper around running a tool — ask for confirmation before a risky
action, log every call, fix up the inputs, or block it outright — without changing the tool's code.

## How It Works

Applied by wrapping a tool in a `MiddlewareAgentTool`; the wrapped tool is registered like any
other, so the agent loop runs the middleware transparently whenever the model calls it. This is the hook
for human-in-the-loop tool approval and tool-level guardrails/observability.

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeAsync(ToolInvocationContext,ToolPipelineDelegate,CancellationToken)` | Processes a tool call, optionally calling `next` to run the tool. |

