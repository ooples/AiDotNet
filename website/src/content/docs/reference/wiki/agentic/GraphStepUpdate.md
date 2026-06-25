---
title: "GraphStepUpdate<TState>"
description: "A streamed update emitted after a single node finishes executing during a graph run: the node's name and the graph state as it stands after that node."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Graph`

A streamed update emitted after a single node finishes executing during a graph run: the node's name
and the graph state as it stands after that node.

## For Beginners

When you stream a graph run, you receive one of these each time a step
completes — letting you watch the state evolve node by node (for progress UIs, logging, or
debugging). The last update's `State` is the final result.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphStepUpdate(String,)` | Initializes a new step update. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NodeName` | Gets the name of the node that just executed. |
| `State` | Gets the graph state after the node executed. |

