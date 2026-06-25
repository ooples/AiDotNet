---
title: "IAgentTool"
description: "An executable tool the model can call: a name, a description, a JSON-schema for its arguments, and an asynchronous invocation entry point."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Tools`

An executable tool the model can call: a name, a description, a JSON-schema for its arguments, and
an asynchronous invocation entry point.

## For Beginners

Think of this as one gadget in the assistant's toolbox. It knows its own
name, what it does, and what inputs it expects (the schema). When the model says "use this gadget
with these inputs", `CancellationToken)` is the code that runs.

## How It Works

This is the runnable counterpart to `AiToolDefinition`. The definition is what the model
is *told* about; an `IAgentTool` is what actually *runs* when the model
requests a call. The orchestration loop matches a model-emitted
`ToolCallContent` to a tool by `Name`, passes the parsed arguments to
`CancellationToken)`, and feeds the `ToolInvocationResult` back as a tool message.

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets the natural-language description that helps the model decide when to call the tool. |
| `Name` | Gets the unique name the model references when requesting this tool. |
| `ParametersSchema` | Gets the JSON Schema object describing this tool's parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeAsync(JObject,CancellationToken)` | Executes the tool with the supplied arguments. |
| `ToDefinition` | Produces the provider-facing `AiToolDefinition` for this tool. |

