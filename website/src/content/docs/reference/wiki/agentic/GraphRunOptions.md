---
title: "GraphRunOptions"
description: "Per-run settings for executing a `CompiledStateGraph`."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Agentic.Graph`

Per-run settings for executing a `CompiledStateGraph`.

## For Beginners

Knobs for a single graph run. The most important one is
`MaxSteps`, which caps how many node executions a run may take before giving up — a
safety net against cycles that never end.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSteps` | Gets or sets the maximum number of node executions allowed in a single run before a `GraphRecursionException` is thrown. |

