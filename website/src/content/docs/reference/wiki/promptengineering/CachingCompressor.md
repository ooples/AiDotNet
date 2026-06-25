---
title: "CachingCompressor"
description: "Wrapper compressor that caches compression results for frequently used prompts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Compression`

Wrapper compressor that caches compression results for frequently used prompts.

## For Beginners

Remembers compressions so the same prompt doesn't need to be compressed twice.

Example:

When to use:

- Wrapping expensive compressors (LLM-based)
- When the same prompts are used repeatedly
- In production systems to reduce API calls

## How It Works

This compressor wraps another compressor and caches the results. When the same
prompt is compressed multiple times, the cached result is returned instead of
re-computing the compression. This is particularly useful for LLM-based compressors
where each compression call is expensive.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CachingCompressor(IPromptCompressor,Int32,Nullable<TimeSpan>,Func<String,Int32>)` | Initializes a new instance of the CachingCompressor class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CacheCount` | Gets the number of items currently in the cache. |
| `CacheHitRatio` | Gets the cache hit ratio (hits / total requests). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddToCache(String,String)` | Adds an entry to the cache, evicting old entries if necessary. |
| `ClearCache` | Clears all entries from the cache. |
| `CompressAsync(String,CompressionOptions,CancellationToken)` | Compresses the prompt asynchronously, using cache if available. |
| `CompressCore(String,CompressionOptions)` | Compresses the prompt, using cache if available. |
| `ComputeCacheKey(String,CompressionOptions)` | Computes a cache key for the prompt and options combination. |
| `EvictOldEntries` | Evicts old entries to make room for new ones. |
| `GetCacheStats` | Gets statistics about the cache. |
| `PurgeExpiredEntries` | Removes expired entries from the cache. |

