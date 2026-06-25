---
title: "DelegateAgentTool"
description: "An `IAgentTool` backed by an ordinary C# method (a delegate or a reflected method on an instance)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Tools`

An `IAgentTool` backed by an ordinary C# method (a delegate or a reflected method on an
instance). The parameter schema is generated automatically, arguments are bound from the model's JSON,
and the return value is serialized back to text.

## For Beginners

Hand this a method like `int Add(int a, int b)` and it becomes a tool:
the model sees inputs `a` and `b`, and when it calls the tool with `{"a":2,"b":3}` this
class converts that JSON into real arguments, runs your method, and turns the result back into text.

## How It Works

This is the bridge from "a method you already have" to "a tool the model can call". Supported return
shapes: `void`, a value, `Task`, or `Task`. A
`CancellationToken` parameter (if present) is supplied by the runtime and hidden from the
model's schema.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DelegateAgentTool(String,String,Delegate)` | Initializes a tool from a delegate. |
| `DelegateAgentTool(String,String,MethodInfo,Object)` | Initializes a tool from a method and optional target instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InvokeCoreAsync(JObject,CancellationToken)` |  |

