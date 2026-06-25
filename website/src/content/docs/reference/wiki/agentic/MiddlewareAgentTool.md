---
title: "MiddlewareAgentTool"
description: "An `IAgentTool` decorator that runs a chain of `IToolMiddleware` around an inner tool's execution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

An `IAgentTool` decorator that runs a chain of `IToolMiddleware` around an inner
tool's execution. Register the wrapped tool like any other and the agent loop applies the middleware
transparently whenever the model calls it — the composition root for tool approval, guardrails, logging,
and caching.

## For Beginners

Wrap a tool with this and a list of filters, and every time the model uses that
tool the filters run first (approve it, log it, fix the inputs) — the model still sees the same tool.

## How It Works

The tool's identity (name, description, schema, definition) is delegated to the inner tool, so wrapping is
invisible to the model — only execution is intercepted. Middleware run in registration order (first =
outermost).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MiddlewareAgentTool(IAgentTool,IReadOnlyList<IToolMiddleware>)` | Initializes a new middleware tool decorator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |
| `ParametersSchema` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeAsync(JObject,CancellationToken)` |  |
| `ToDefinition` |  |

