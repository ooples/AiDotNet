---
title: "FunctionAgentTool"
description: "An `IAgentTool` whose parameter schema is supplied up front and whose execution is a caller-provided delegate — i.e."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Tools`

An `IAgentTool` whose parameter schema is supplied up front and whose execution is a
caller-provided delegate — i.e. no reflection at invoke time. This is the target type for the
source generator, which emits a precomputed schema and a typed argument binder for each
`AgentToolAttribute`-annotated method.

## For Beginners

This is a tool whose "what inputs do I take" (schema) and "what do I do"
(the invoker) are both handed in ready-made, so nothing has to be figured out by reflection while the
agent runs.

## How It Works

Compared to `DelegateAgentTool` (which inspects a method via reflection at construction
and invocation), this type carries an already-built schema and a ready-to-run invoker, making it
AOT-friendly and allocation-light. You rarely construct it by hand; the generated
`CreateAgentTools()` extension produces these for you.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FunctionAgentTool(String,String,JObject,Func<JObject,CancellationToken,Task<String>>)` | Initializes a new function-backed tool. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeCoreAsync(JObject,CancellationToken)` |  |

