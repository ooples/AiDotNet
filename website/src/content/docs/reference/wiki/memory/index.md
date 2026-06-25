---
title: "Memory"
description: "All 10 public types in the AiDotNet.memory namespace, organized by kind."
section: "API Reference"
---

**10** public types in this namespace, organized by kind.

## Models & Types (6)

| Type | Summary |
|:-----|:--------|
| [`InferenceContext<T>`](/docs/reference/wiki/memory/inferencecontext/) | Provides a scoped context for inference operations with automatic tensor pooling and lifecycle management. |
| [`LayerWorkspace<T>`](/docs/reference/wiki/memory/layerworkspace/) | Per-layer workspace that manages pre-allocated tensor buffers for zero-allocation forward passes. |
| [`PoolStatistics`](/docs/reference/wiki/memory/poolstatistics/) | Provides statistics about the current state of a tensor pool. |
| [`PooledMemoryOwner<T>`](/docs/reference/wiki/memory/pooledmemoryowner/) | An IMemoryOwner implementation that returns memory to the pool when disposed. |
| [`PooledTensor<T>`](/docs/reference/wiki/memory/pooledtensor/) | A RAII wrapper that automatically returns a pooled tensor to its pool when disposed. |
| [`TensorPool<T>`](/docs/reference/wiki/memory/tensorpool/) | A high-performance, thread-safe memory pool for reusing tensors during neural network operations. |

## Structs (2)

| Type | Summary |
|:-----|:--------|
| [`InferenceScopeHandle<T>`](/docs/reference/wiki/memory/inferencescopehandle/) | A disposable handle that restores the previous inference context when disposed. |
| [`ShapeKey`](/docs/reference/wiki/memory/shapekey/) | Value-type shape key for arena dictionary lookups. |

## Options & Configuration (1)

| Type | Summary |
|:-----|:--------|
| [`PoolingOptions`](/docs/reference/wiki/memory/poolingoptions/) | Configuration options for the tensor pool, which manages memory reuse during neural network operations. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`InferenceScope<T>`](/docs/reference/wiki/memory/inferencescope/) | Provides ambient (thread-local) context support for `InferenceContext`. |

