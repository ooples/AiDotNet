---
title: "MemoryCacheEvictionPolicy"
description: "Policy for evicting items from the in-memory cache when it's full."
section: "API Reference"
---

`Enums` · `AiDotNet.Data.Loaders`

Policy for evicting items from the in-memory cache when it's full.

## Fields

| Field | Summary |
|:-----|:--------|
| `FIFO` | First In First Out: evict the oldest cached item. |
| `LFU` | Least Frequently Used: evict the item accessed fewest times. |
| `LRU` | Least Recently Used: evict the item accessed longest ago. |

