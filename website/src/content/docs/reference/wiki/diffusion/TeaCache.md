---
title: "TeaCache<T>"
description: "Timestep Embedding Aware Cache (TeaCache) for accelerating video diffusion inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Acceleration`

Timestep Embedding Aware Cache (TeaCache) for accelerating video diffusion inference.

## For Beginners

TeaCache (Timestep Embedding Aware Cache) accelerates DiT-based video generation by caching and reusing intermediate computations when timestep embeddings are similar. It provides significant speedup with minimal quality loss.

## How It Works

**References:**

- Paper: "TeaCache: Timestep-Aware KV-Cache for Efficient Video Diffusion" (2025)

TeaCache accelerates video diffusion inference by caching and reusing key-value pairs
across denoising timesteps. The key insight is that adjacent timesteps have very similar
attention patterns, so KV pairs can be reused with minimal quality degradation.

The cache uses a similarity threshold to determine when to recompute:

- If the timestep embedding change is small, reuse cached KV pairs
- If the change exceeds the threshold, recompute attention
- Achieves 2-3x speedup on typical 50-step diffusion sampling

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TeaCache(Double,Int32)` | Initializes a new TeaCache. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CacheHits` | Gets the number of cache hits. |
| `CacheMisses` | Gets the number of cache misses. |
| `HitRate` | Gets the cache hit rate. |
| `MaxCacheSize` | Gets the maximum cache size. |
| `ReuseThreshold` | Gets the reuse threshold for timestep embedding similarity. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Reset` | Resets the cache and statistics. |
| `Retrieve(String)` | Retrieves cached KV pairs. |
| `ShouldReuse(String,Double)` | Checks if cached KV pairs can be reused for the given layer and timestep. |
| `Store(String,Tensor<>,Double)` | Stores KV pairs in the cache. |

