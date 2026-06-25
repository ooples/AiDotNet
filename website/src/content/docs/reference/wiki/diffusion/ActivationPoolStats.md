---
title: "ActivationPoolStats"
description: "Statistics about activation pool usage."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Memory`

Statistics about activation pool usage.
Thread-safe counters for concurrent access.

## Properties

| Property | Summary |
|:-----|:--------|
| `CacheHits` | Number of times a tensor was found in the pool. |
| `CacheMisses` | Number of times a new tensor had to be allocated. |
| `Evictions` | Number of tensors evicted due to memory pressure. |
| `HitRatio` | Cache hit ratio (0-1). |
| `PooledTensors` | Current number of tensors in the pool. |
| `Returns` | Number of tensors returned to the pool. |

## Methods

| Method | Summary |
|:-----|:--------|
| `IncrementCacheHits` | Thread-safe increment of cache hits counter. |
| `IncrementCacheMisses` | Thread-safe increment of cache misses counter. |
| `IncrementEvictions` | Thread-safe increment of evictions counter. |
| `IncrementReturns` | Thread-safe increment of returns counter. |
| `ToString` |  |

