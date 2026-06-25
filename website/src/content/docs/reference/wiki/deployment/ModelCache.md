---
title: "ModelCache<T>"
description: "Cache for model inference results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Runtime`

Cache for model inference results.

## For Beginners

ModelCache provides AI safety functionality. Default values follow the original paper settings.

## Methods

| Method | Summary |
|:-----|:--------|
| `Clear` | Clears the cache. |
| `EvictLFU(Int32)` | Evicts least frequently used entries. |
| `EvictLRU(Int32)` | Evicts least recently used entries to maintain size limit. |
| `EvictOlderThan(TimeSpan)` | Removes entries older than the specified age. |
| `Get(String,[])` | Gets a cached result for the given input. |
| `GetStatistics` | Gets cache statistics. |
| `Put(String,[],[])` | Puts a result in the cache. |

