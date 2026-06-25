---
title: "PoolingOptions"
description: "Configuration options for the tensor pool, which manages memory reuse during neural network operations."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Memory`

Configuration options for the tensor pool, which manages memory reuse during neural network operations.
The tensor pool helps reduce memory allocations and garbage collection pressure by reusing tensor buffers.

## How It Works

Tensor pooling is especially beneficial for:

- Inference operations with consistent input sizes
- Training loops where tensor shapes are predictable
- High-throughput scenarios where allocation overhead matters

Example usage:

## Properties

| Property | Summary |
|:-----|:--------|
| `Enabled` | Gets or sets a value indicating whether tensor pooling is enabled. |
| `MaxElementsToPool` | Gets or sets the maximum number of elements a single tensor can have to be eligible for pooling. |
| `MaxItemsPerBucket` | Gets or sets the maximum number of tensors to keep in each shape bucket. |
| `MaxPoolSizeBytes` | Gets or sets the maximum memory size of the tensor pool in bytes. |
| `MaxPoolSizeMB` | Gets or sets the maximum memory size of the tensor pool in megabytes. |
| `UseWeakReferences` | Gets or sets a value indicating whether to use weak references for pooled tensors. |

