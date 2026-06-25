---
title: "ToolResultContent"
description: "The result of executing a tool, fed back to the model to continue a tool-calling turn."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models`

The result of executing a tool, fed back to the model to continue a tool-calling turn.

## For Beginners

This is the answer slip you hand back after running the tool the model
asked for. It carries the same ticket number (`CallId`) as the request, the tool's
output (`Result`), and a flag (`IsError`) so the model knows whether the
tool succeeded or failed.

## How It Works

After the assistant emits a `ToolCallContent` and the caller runs the tool, the output
is returned to the model as a `Tool` message containing this content. The
`CallId` must match the originating call so the model knows which request this answers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ToolResultContent(String,String,Boolean)` | Initializes a new tool result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CallId` | Gets the id of the tool call this result corresponds to. |
| `IsError` | Gets a value indicating whether the tool invocation failed. |
| `Result` | Gets the tool's output as a string (plain text or serialized JSON). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` |  |

