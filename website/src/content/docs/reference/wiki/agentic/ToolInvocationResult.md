---
title: "ToolInvocationResult"
description: "The outcome of executing an `IAgentTool`: the text fed back to the model plus a flag indicating whether the tool failed."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Tools`

The outcome of executing an `IAgentTool`: the text fed back to the model plus a flag
indicating whether the tool failed.

## For Beginners

When a tool runs, it produces an answer string. This wraps that answer
together with a yes/no flag for "did it go wrong?". Use `String)` for a good result and
`String)` for a failure message.

## How It Works

Tools return results rather than throwing for expected failures, so the model can see the error
and recover (for example, retry with different arguments). The `Content` is what gets
placed into the `ToolResultContent` sent back to the model.

## Properties

| Property | Summary |
|:-----|:--------|
| `Content` | Gets the result text to feed back to the model (plain text or serialized JSON). |
| `IsError` | Gets a value indicating whether the tool invocation failed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Error(String)` | Creates a failed result carrying an error message. |
| `Success(String)` | Creates a successful result. |

