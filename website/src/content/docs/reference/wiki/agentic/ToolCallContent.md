---
title: "ToolCallContent"
description: "A request, emitted by the assistant, to invoke a named tool/function with JSON arguments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models`

A request, emitted by the assistant, to invoke a named tool/function with JSON arguments.

## For Beginners

When the model wants to use a tool, it doesn't run the tool itself —
it hands you a filled-in request form: "call the tool named `get_weather` with
`{\"city\":\"Paris\"}`, and here's a ticket number so we can match up the answer." This class
is that form. The ticket number is `CallId`.

## How It Works

This is the heart of native function calling. Instead of the model writing "please run the
calculator" as prose for us to parse, the provider returns a structured tool-call: a stable
`CallId`, the `ToolName` to invoke, and the `ArgumentsJson`
the model chose. The caller executes the tool and replies with a matching
`ToolResultContent` carrying the same `CallId`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ToolCallContent(String,String,String)` | Initializes a new tool-call request. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ArgumentsJson` | Gets the model-chosen arguments as a raw JSON object string (never `null`; defaults to `{}`). |
| `CallId` | Gets the provider-assigned id correlating this call to its `ToolResultContent`. |
| `ToolName` | Gets the name of the tool the model requested. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` |  |

