---
title: "GraphRecursionException"
description: "Thrown when a graph run exceeds its configured maximum number of steps, which usually indicates a cycle that never reaches the end node."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Agentic.Graph`

Thrown when a graph run exceeds its configured maximum number of steps, which usually indicates a
cycle that never reaches the end node.

## For Beginners

Graphs can loop (a node can route back to an earlier node). To stop an
accidental infinite loop, every run has a step budget. If the graph keeps going past that budget,
this exception is raised so you can fix the routing or raise the limit deliberately.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphRecursionException(Int32)` | Initializes a new instance of the `GraphRecursionException` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSteps` | Gets the step budget that was exceeded. |

