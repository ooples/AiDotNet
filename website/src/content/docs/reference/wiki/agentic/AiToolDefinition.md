---
title: "AiToolDefinition"
description: "Describes a tool/function the model is allowed to call: its name, a description, and a JSON-schema describing its parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models`

Describes a tool/function the model is allowed to call: its name, a description, and a JSON-schema
describing its parameters.

## For Beginners

Before a model can use a tool, you have to describe the tool to it — like
a menu entry: the tool's name, what it does, and what inputs it needs. This class is that menu entry.
The "what inputs it needs" part is written in JSON Schema (e.g., "an object with a string field
called `city`"). In later phases this schema is generated automatically from your C# method
signature, so you won't usually hand-write it.

## How It Works

This is the *declaration* sent to the model (distinct from the executable tool in the Tools
layer). The model uses the name and description to decide *whether* to call the tool, and the
parameter schema to decide *how* to fill in the arguments. The schema is a standard JSON Schema
object (the same shape OpenAI, Anthropic, and the local engine all consume).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AiToolDefinition(String,String,JObject)` | Initializes a new tool definition. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the natural-language description that helps the model decide when to call the tool. |
| `Name` | Gets the unique name the model references when requesting this tool. |
| `ParametersSchema` | Gets the JSON Schema object describing the tool's parameters. |

