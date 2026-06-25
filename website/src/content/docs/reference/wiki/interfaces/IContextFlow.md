---
title: "IContextFlow<T>"
description: "Interface for Context Flow mechanism - maintains distinct information pathways and update rates for each nested optimization level."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for Context Flow mechanism - maintains distinct information pathways
and update rates for each nested optimization level.
Core component of nested learning paradigm.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfLevels` | Gets the number of context flow levels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompressContext(Vector<>,Int32)` | Compresses internal context flows (deep learning compression mechanism). |
| `ComputeContextGradients(Vector<>,Int32)` | Computes gradients with respect to context flow for backpropagation. |
| `GetContextState(Int32)` | Gets the current context state for a specific optimization level. |
| `PropagateContext(Vector<>,Int32)` | Propagates context through the flow network at a specific optimization level. |
| `Reset` | Resets the context flow to initial state. |
| `UpdateFlow(Vector<>[],[])` | Updates the context flow based on multi-level optimization. |

