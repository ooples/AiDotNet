---
title: "AgentToolAttribute"
description: "Marks a method as an agent tool so it can be discovered and exposed to a model with an auto-generated JSON schema."
section: "API Reference"
---

`Attributes` · `AiDotNet.Agentic.Tools`

Marks a method as an agent tool so it can be discovered and exposed to a model with an
auto-generated JSON schema.

## For Beginners

Instead of hand-writing a tool class, you can write a normal C# method
(e.g., `GetWeather(string city)`), put `[AgentTool("Gets current weather")]` on it, and
the library figures out the tool's name, description, and input schema for you.

## How It Works

Annotate a public method, then register the containing object with the tool layer; the method's
signature is turned into an `AiToolDefinition` automatically.
The tool name defaults to the method name when `Name` is not set.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AgentToolAttribute(String)` | Initializes the attribute. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the natural-language description of the tool. |
| `Name` | Gets or sets the tool name exposed to the model. |

