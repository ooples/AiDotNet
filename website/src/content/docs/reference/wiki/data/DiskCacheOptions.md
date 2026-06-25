---
title: "DiskCacheOptions"
description: "Configuration options for disk-based pipeline caching."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Pipeline`

Configuration options for disk-based pipeline caching.

## For Beginners

These settings control where processed data is saved
and how much disk space it can use. Once data is cached, subsequent training epochs
can skip expensive preprocessing.

## How It Works

Controls how pipeline snapshots are stored on disk, including cache location,
size limits, and eviction policies.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoInvalidateOnSourceChange` | Gets or sets whether to automatically invalidate the cache when source data changes. |
| `CacheDirectory` | Gets or sets the directory where cache files are stored. |
| `CompressData` | Gets or sets whether to compress cached data. |
| `EvictionPolicy` | Gets or sets the eviction policy when cache is full. |
| `MaxAge` | Gets or sets the maximum age of cache entries before they are considered stale. |
| `MaxCacheSizeBytes` | Gets or sets the maximum total cache size in bytes. |
| `VerifyIntegrity` | Gets or sets whether to verify cache integrity on load using checksums. |

