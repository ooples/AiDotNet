---
title: "GpuExecutionModeConfig"
description: "GPU execution mode controlling how operations are scheduled and executed."
section: "API Reference"
---

`Enums` · `AiDotNet.Engines`

GPU execution mode controlling how operations are scheduled and executed.

## For Beginners

This controls how GPU operations are executed:

- **Auto**: Automatically select best mode based on GPU capabilities (recommended)
- **Eager**: Execute each operation immediately (most compatible, simplest debugging)
- **Deferred**: Batch operations for optimization (highest performance, 10-50x faster)
- **ScopedDeferred**: Batch within explicit scopes (balanced performance and control)

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Automatically select best execution mode based on GPU capabilities. |
| `Deferred` | Deferred execution - operations are recorded and executed as optimized graphs. |
| `Eager` | Eager execution - each operation runs immediately and synchronously. |
| `ScopedDeferred` | Scoped deferred execution - operations within explicit scopes are batched. |

