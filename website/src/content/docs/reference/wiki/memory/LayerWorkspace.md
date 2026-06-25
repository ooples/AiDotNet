---
title: "LayerWorkspace<T>"
description: "Per-layer workspace that manages pre-allocated tensor buffers for zero-allocation forward passes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Memory`

Per-layer workspace that manages pre-allocated tensor buffers for zero-allocation forward passes.
Uses index-based access for maximum performance — array lookup instead of dictionary.

Usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayerWorkspace(Int32,Int32)` | Create a workspace with the specified number of timestep and sequence buffer slots. |

## Properties

| Property | Summary |
|:-----|:--------|
| `TotalPreAllocated` | Gets the total number of pre-allocated tensors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BeginForward(Int32,Int32)` | Prepare for a forward pass. |
| `DeclareSequence(Int32,Int32[])` | Declare a per-sequence buffer at the given index. |
| `DeclareTimestep(Int32,Int32[])` | Declare a per-timestep buffer at the given index. |
| `EndForward` | End the forward pass. |
| `Sequence(Int32)` | Get a pre-allocated sequence buffer by index. |
| `Timestep(Int32)` | Get a pre-allocated timestep buffer by index. |

