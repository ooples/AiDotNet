---
title: "StreamingToolCallUpdate"
description: "An incremental fragment of a tool call arriving over a streaming response."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models`

An incremental fragment of a tool call arriving over a streaming response.

## For Beginners

While streaming, a tool call is delivered a little at a time — first
"I'm going to call tool #0 named `get_weather`", then the arguments dribble in like
`{"ci`, `ty":"Par`, `is"}`. The `Index` tells you which tool call a
fragment belongs to so you can stitch the right pieces together.

## How It Works

When a model streams a tool call, the pieces arrive across several chunks: the id and tool name
usually come first, then the JSON arguments stream in fragments that must be concatenated by
`Index`. Accumulating these fragments yields a complete `ToolCallContent`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StreamingToolCallUpdate(Int32,String,String,String)` | Initializes a new streaming tool-call fragment. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ArgumentsJsonFragment` | Gets a fragment of the arguments JSON to append, or `null` when this fragment carries none. |
| `CallId` | Gets the tool-call id, or `null` when this fragment does not carry it. |
| `Index` | Gets the zero-based index identifying which tool call this fragment contributes to. |
| `ToolName` | Gets the tool name, or `null` when this fragment does not carry it. |

