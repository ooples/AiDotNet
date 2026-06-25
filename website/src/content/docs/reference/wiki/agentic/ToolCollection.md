---
title: "ToolCollection"
description: "A named set of executable tools: registers `IAgentTool` instances, exposes their `AiToolDefinition`s for a chat request, and dispatches model tool-calls to the right tool."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Tools`

A named set of executable tools: registers `IAgentTool` instances, exposes their
`AiToolDefinition`s for a chat request, and dispatches model tool-calls to the right tool.

## For Beginners

Think of this as the labeled toolbox you give the model. It can list what's
inside (so the model knows its options) and, when the model asks to use a specific tool, it finds that
tool, runs it, and packages the answer to hand back to the model.

## How It Works

This is the bridge between the model and your code during a tool-calling turn. Register tools, hand
`GetDefinitions` to `Tools`, and when the model replies with
`ToolCallContent`, call `CancellationToken)` to run the tool and produce
the `Tool` message to append before the next model call.

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of registered tools. |
| `Tools` | Gets the registered tools. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IAgentTool)` | Registers a tool. |
| `AddDelegate(String,String,Delegate)` | Creates a tool from a delegate and registers it. |
| `AddFrom(Object)` | Scans an object for `AgentToolAttribute`-annotated methods and registers each as a tool. |
| `Contains(String)` | Determines whether a tool with the given name is registered. |
| `Get(String)` | Gets a tool by name. |
| `GetDefinitions` | Produces the provider-facing definitions for all registered tools, for use as `Tools`. |
| `InvokeAsync(ToolCallContent,CancellationToken)` | Executes a model-emitted tool call and returns the raw result. |
| `InvokeToToolMessageAsync(ToolCallContent,CancellationToken)` | Executes a model-emitted tool call and wraps the result in a `Tool` message ready to append to the conversation before the next model call. |

