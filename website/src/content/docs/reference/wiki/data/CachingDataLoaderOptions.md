---
title: "CachingDataLoaderOptions"
description: "Configuration options for caching data loader."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Loaders`

Configuration options for caching data loader.

## Properties

| Property | Summary |
|:-----|:--------|
| `DiskCacheDirectory` | Directory for disk cache files. |
| `EnableDiskCache` | Whether to cache on disk as well (two-level cache). |
| `EvictionPolicy` | Cache eviction policy. |
| `MaxCacheSize` | Maximum number of batches to cache in memory. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

