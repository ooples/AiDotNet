---
title: "PooledMemoryOwner<T>"
description: "An IMemoryOwner implementation that returns memory to the pool when disposed."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Memory`

An IMemoryOwner implementation that returns memory to the pool when disposed.

## Properties

| Property | Summary |
|:-----|:--------|
| `Array` | Gets the underlying array (internal use for pool return). |
| `Memory` | Gets the Memory wrapped by this owner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Returns the memory to the pool and releases the reference. |

