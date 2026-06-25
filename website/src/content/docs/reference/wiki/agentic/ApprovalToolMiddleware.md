---
title: "ApprovalToolMiddleware"
description: "An `IToolMiddleware` that gates tool execution behind an approval check — human-in-the-loop or policy-based tool authorization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

An `IToolMiddleware` that gates tool execution behind an approval check — human-in-the-loop or
policy-based tool authorization. When the check denies a call, the tool does not run and a deny result is
returned to the model so it can react.

## For Beginners

A permission gate for tools. Before a (say) "delete file" or "send email" tool
runs, your approval function decides yes/no. On "no", the tool is skipped and the model is told it wasn't
allowed. Wrap sensitive tools with this to keep a human (or a rule) in control.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ApprovalToolMiddleware(Func<ToolInvocationContext,Boolean>,String)` | Initializes a new approval middleware. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeAsync(ToolInvocationContext,ToolPipelineDelegate,CancellationToken)` |  |

