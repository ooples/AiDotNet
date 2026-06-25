---
title: "CachingDataLoader<TKey, TValue>"
description: "Wraps data loading with an in-memory cache to avoid redundant I/O."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Wraps data loading with an in-memory cache to avoid redundant I/O.

## How It Works

Caches loaded batches by their key (typically batch index) to avoid reloading
data that has already been seen. Useful when iterating over the same data
multiple epochs or when random access patterns cause repeated reads.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CachingDataLoader(CachingDataLoaderOptions)` | Creates a new caching data loader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of items currently in the cache. |
| `HitRatio` | Gets the cache hit ratio (hits / total requests). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Clears all cached items and resets statistics. |
| `Contains()` | Checks if a key is in the cache. |
| `GetOrLoad(,Func<,>)` | Gets a value from the cache, or loads it using the provided factory. |
| `Invalidate()` | Invalidates a specific cache entry. |

