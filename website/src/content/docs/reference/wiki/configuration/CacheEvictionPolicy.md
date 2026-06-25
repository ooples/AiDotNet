---
title: "CacheEvictionPolicy"
description: "Cache eviction policies for KV cache management."
section: "API Reference"
---

`Enums` · `AiDotNet.Configuration`

Cache eviction policies for KV cache management.

## Fields

| Field | Summary |
|:-----|:--------|
| `FIFO` | First In First Out - evicts oldest entries first. |
| `LFU` | Least Frequently Used - evicts entries with lowest access count. |
| `LRU` | Least Recently Used - evicts entries that haven't been accessed recently. |

