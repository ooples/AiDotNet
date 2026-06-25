---
title: "ActivationPool<T>"
description: "Memory pool for tensor activations during diffusion model forward/backward passes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Memory`

Memory pool for tensor activations during diffusion model forward/backward passes.

## For Beginners

When running diffusion models, temporary data (activations)
is created at each layer.

Without pooling:

- Layer 1 creates activation A (allocate new memory)
- Layer 2 creates activation B (allocate new memory)
- Layer 1's activation A becomes garbage (GC must clean up)
- This creates memory pressure and GC pauses

With pooling:

- Layer 1 borrows a buffer from the pool
- Layer 1 returns the buffer when done
- Layer 2 reuses the same buffer
- No garbage, no GC pauses, faster inference

Usage:
```cs
using var pool = new ActivationPool<float>(maxMemoryMB: 2048);

// During forward pass
var activation = pool.Rent(new[] { 1, 256, 64, 64 });
// ... use activation ...
pool.Return(activation);
```

## How It Works

Diffusion models process large tensors through many layers, creating significant
memory pressure from intermediate activations. This pool reduces allocations
by recycling tensor buffers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ActivationPool(Int64)` | Initializes a new activation pool with specified memory limit. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Stats` | Statistics about pool usage. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateTotalElements(Int32[])` | Calculates total elements in a shape. |
| `Clear` | Clears all pooled tensors and resets memory accounting. |
| `Dispose` | Disposes the pool and releases all tensors. |
| `EvictOldest(Int64)` | Evicts oldest tensors to make room for new allocation. |
| `GetMemorySize(Int64)` | Estimates memory size for a given element count. |
| `GetMemoryUsage` | Gets current memory usage statistics. |
| `GetSizeClass(Int64)` | Gets the size class bucket for a given element count. |
| `Rent(Int32[])` | Rents a tensor from the pool or creates a new one. |
| `Return(Tensor<>)` | Returns a tensor to the pool for reuse. |
| `ShapeMatches(Int32[],Int32[])` | Checks if two shapes match exactly. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_currentMemoryBytes` | Current estimated memory usage. |
| `_disposed` | Whether the pool has been disposed. |
| `_maxMemoryBytes` | Maximum memory to use for pooling in bytes. |
| `_memoryLock` | Lock for memory accounting. |
| `_pools` | Size class buckets for tensor pooling. |

