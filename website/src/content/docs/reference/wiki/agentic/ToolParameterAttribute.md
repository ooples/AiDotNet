---
title: "ToolParameterAttribute"
description: "Adds a description (and optional required-override) to a tool method parameter, enriching the auto-generated JSON schema."
section: "API Reference"
---

`Attributes` · `AiDotNet.Agentic.Tools`

Adds a description (and optional required-override) to a tool method parameter, enriching the
auto-generated JSON schema.

## For Beginners

Put this on a tool method's parameter to tell the model what that input
means — e.g. `[ToolParameter("City name, e.g. 'Paris'")] string city`. Clearer descriptions
lead to more accurate tool calls.

## How It Works

Parameter descriptions materially improve how well a model fills in tool arguments. By default a
parameter is required unless it has a default value or is nullable; set `Required` to
override that inference.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ToolParameterAttribute(String)` | Initializes the attribute. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the natural-language description of the parameter. |
| `Required` | Gets or sets an explicit override for whether the parameter is required. |

