---
title: "CacheConfig"
description: "Configuration for model caching - storing loaded models in memory to avoid repeated loading."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Configuration`

Configuration for model caching - storing loaded models in memory to avoid repeated loading.

## For Beginners

Loading an AI model from disk takes time. Caching keeps recently-used
models in memory so they can be used again instantly, like keeping your frequently-used apps
open on your phone instead of closing and reopening them.

Benefits:

- Much faster inference (no model loading time)
- Better throughput when serving multiple requests
- Reduces disk I/O

Considerations:

- Uses memory (RAM) to store models
- Limited cache size - old models get evicted when full

Eviction Policies (what to remove when cache is full):

- LRU (Least Recently Used): Removes models you haven't used in a while (recommended)
- LFU (Least Frequently Used): Removes models used least often
- FIFO: Removes oldest models first
- Random: Removes random models (simple but unpredictable)

For most applications, LRU with a moderate max size works well.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultTTL` | Gets or sets the cache entry time-to-live as a TimeSpan (default: 1 hour). |
| `Enabled` | Gets or sets whether caching is enabled (default: true). |
| `EvictionPolicy` | Gets or sets the cache eviction policy (default: LRU). |
| `MaxCacheSize` | Gets or sets the maximum number of models to cache (default: 10). |
| `MaxSizeMB` | Gets or sets the maximum cache size in megabytes (default: 100.0 MB). |
| `PreloadModels` | Gets or sets whether to preload models on startup (default: false). |
| `TimeToLiveSeconds` | Gets or sets the cache entry time-to-live in seconds (default: 3600 = 1 hour). |
| `TrackStatistics` | Gets or sets whether to track cache hit/miss statistics (default: true). |

