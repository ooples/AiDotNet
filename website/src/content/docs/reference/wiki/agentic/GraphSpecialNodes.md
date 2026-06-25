---
title: "GraphSpecialNodes"
description: "Reserved node names recognized by the graph runtime."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Graph`

Reserved node names recognized by the graph runtime.

## For Beginners

A graph finishes when flow reaches the special `End` node.
You route to it with `AddEdge("lastNode", GraphSpecialNodes.End)` (or return it from a
conditional router) to say "we're done".

## Fields

| Field | Summary |
|:-----|:--------|
| `End` | The terminal node name. |

