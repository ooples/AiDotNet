---
title: "AgentToolFactory"
description: "Creates `IAgentTool` instances from delegates or from objects whose methods are annotated with `AgentToolAttribute`."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Tools`

Creates `IAgentTool` instances from delegates or from objects whose methods are annotated
with `AgentToolAttribute`.

## For Beginners

Two easy ways to make tools without writing a tool class:

- `Delegate)`: wrap a lambda or method reference directly.
- `Object)`: take any object, find every method you marked with `[AgentTool]`,

and turn each into a tool automatically.

## Methods

| Method | Summary |
|:-----|:--------|
| `FromDelegate(String,String,Delegate)` | Creates a tool from a delegate (lambda or method group). |
| `ScanInstance(Object)` | Scans an object for methods annotated with `AgentToolAttribute` and creates a tool for each (covering both instance and static annotated methods). |

