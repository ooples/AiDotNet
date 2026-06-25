---
title: "ToolInvocationContext"
description: "The mutable state flowing through a tool-invocation middleware pipeline: which tool is being called, the (rewritable) arguments, and a shared property bag."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

The mutable state flowing through a tool-invocation middleware pipeline: which tool is being called, the
(rewritable) arguments, and a shared property bag.

## For Beginners

The request slip for running a tool. Middleware can read which tool was asked
for and with what inputs, tweak the inputs, or stash notes — before the tool actually runs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ToolInvocationContext(String,JObject)` | Initializes a new tool-invocation context. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Arguments` | Gets or sets the tool arguments (middleware may rewrite them before execution). |
| `Items` | Gets a property bag for sharing state between middleware stages. |
| `ToolName` | Gets the name of the tool being invoked. |

