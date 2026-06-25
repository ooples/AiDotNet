---
title: "TensorPool<T>"
description: "A high-performance, thread-safe memory pool for reusing tensors during neural network operations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Memory`

A high-performance, thread-safe memory pool for reusing tensors during neural network operations.
Reduces memory allocations and garbage collection pressure by pooling tensor buffers.

## How It Works

The tensor pool maintains buckets of pre-allocated tensors grouped by shape.
When a tensor is requested via `Int32[])`, the pool returns an existing tensor
from the appropriate bucket if available, otherwise allocates a new one.

When tensors are returned via `Tensor{`, they are cleared and added
back to the pool for future reuse, up to the configured memory limits.

Basic usage example:

For automatic lifetime management, use `Int32[])` which returns
a `PooledTensor` that automatically returns itself when disposed:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TensorPool` | Initializes a new instance of the `TensorPool` class with default options. |
| `TensorPool(Int32)` | Initializes a new instance of the `TensorPool` class with the specified maximum pool size. |
| `TensorPool(PoolingOptions)` | Initializes a new instance of the `TensorPool` class with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentMemoryBytes` | Gets the current memory usage of all pooled tensors, in bytes. |
| `CurrentPoolSizeBytes` | Gets the current size of the pool in bytes. |
| `MaxPoolSizeBytes` | Gets the maximum allowed memory size for the pool, in bytes. |
| `Options` | Gets the pooling options configured for this pool. |
| `TotalPooledTensors` | Gets the total number of tensors currently held in the pool across all buckets. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Removes all tensors and memory from the pool and resets memory tracking. |
| `Dispose` | Disposes the tensor pool and releases all pooled tensors. |
| `GetStatistics` | Gets current statistics about the pool's state. |
| `Rent(Int32[])` | Rents a tensor with the specified shape from the pool. |
| `RentMemory(Int32)` | Rents raw memory from the pool for lower-level operations. |
| `RentPooled(Int32[])` | Rents a tensor wrapped in a `PooledTensor` for automatic pool return on disposal. |
| `Return(Tensor<>)` | Returns a tensor to the pool for future reuse. |
| `ReturnMemory([])` | Returns memory to the pool for future reuse. |

