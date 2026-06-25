---
title: "AgentToolBase"
description: "Base class for tools that implements the common metadata plumbing, leaving only the behavior (`CancellationToken)`) for subclasses."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Agentic.Tools`

Base class for tools that implements the common metadata plumbing, leaving only the behavior
(`CancellationToken)`) for subclasses.

## For Beginners

Inherit from this to make a custom tool. You provide the name, the
description, the input schema, and one method that does the work. The base class handles the rest,
including turning an accidental crash into a clean "this tool failed" message the model can read.

## How It Works

Follows the template-method pattern used across AiDotNet: this base validates and stores the name,
description, and parameter schema, and builds the `AiToolDefinition`; subclasses supply
only the execution logic. `CancellationToken)` wraps `CancellationToken)` so unexpected
exceptions become a failed `ToolInvocationResult` instead of crashing the agent loop.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AgentToolBase(String,String,JObject)` | Initializes the tool's metadata. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |
| `ParametersSchema` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeAsync(JObject,CancellationToken)` |  |
| `InvokeCoreAsync(JObject,CancellationToken)` | Executes the tool's behavior. |
| `ToDefinition` |  |

