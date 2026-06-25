---
title: "EpisodeCache<T, TInput, TOutput>"
description: "Caches sampled episodes for reuse, reducing the cost of repeated episode generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

Caches sampled episodes for reuse, reducing the cost of repeated episode generation.
Useful when episode sampling is expensive (e.g., loading from disk or complex preprocessing).
Supports LRU eviction when the cache exceeds a configured capacity.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EpisodeCache(Int32)` | Creates an episode cache with the specified capacity. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Capacity` | Gets the maximum number of episodes the cache can hold. |
| `Count` | Gets the current number of cached episodes. |
| `HitCount` | Gets the total number of cache hits since creation. |
| `HitRate` | Gets the cache hit rate as a value in [0, 1]. |
| `MissCount` | Gets the total number of cache misses since creation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Clears all cached episodes and resets hit/miss counters. |
| `Put(IEpisode<,,>)` | Adds an episode to the cache. |
| `PutAll(IEnumerable<IEpisode<,,>>)` | Adds multiple episodes to the cache. |
| `TryGet(Int32,IEpisode<,,>)` | Tries to retrieve a cached episode by its ID. |

