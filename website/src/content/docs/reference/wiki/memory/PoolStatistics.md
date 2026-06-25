---
title: "PoolStatistics"
description: "Provides statistics about the current state of a tensor pool."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Memory`

Provides statistics about the current state of a tensor pool.
Use this class to monitor pool usage and tune pooling parameters.

## How It Works

Pool statistics help you understand:

- How much memory is currently being used by pooled tensors
- How many tensors are available for reuse
- Whether the pool is operating efficiently

Example usage:

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentMemoryBytes` | Gets or sets the total memory currently used by pooled tensors, in bytes. |
| `MaxMemoryBytes` | Gets or sets the maximum memory allowed for pooling, in bytes. |
| `MemoryUtilizationPercent` | Gets the percentage of maximum pool memory currently in use. |
| `PooledTensorCount` | Gets or sets the number of tensors currently held in the pool. |
| `TensorBuckets` | Gets or sets the number of shape buckets in the pool. |

