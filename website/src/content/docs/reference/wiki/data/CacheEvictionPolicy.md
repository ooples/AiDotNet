---
title: "CacheEvictionPolicy"
description: "Policy for evicting cache entries when the cache is full."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Pipeline`

Policy for evicting cache entries when the cache is full.

## Fields

| Field | Summary |
|:-----|:--------|
| `LargestFirst` | Remove the largest entries first. |
| `LeastRecentlyUsed` | Remove the least recently used (accessed) entries first. |
| `OldestFirst` | Remove the oldest entries first (by creation time). |

