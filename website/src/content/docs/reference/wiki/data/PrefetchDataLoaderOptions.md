---
title: "PrefetchDataLoaderOptions"
description: "Configuration options for prefetch-enabled data loading."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Loaders`

Configuration options for prefetch-enabled data loading.

## Properties

| Property | Summary |
|:-----|:--------|
| `PrefetchCount` | Number of batches to prefetch ahead. |
| `TimeoutMs` | Timeout in milliseconds for waiting on a prefetched batch. |
| `UseBackgroundThread` | Whether to use a background thread for prefetching. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

