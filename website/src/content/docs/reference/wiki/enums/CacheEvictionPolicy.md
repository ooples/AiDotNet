---
title: "CacheEvictionPolicy"
description: "Cache eviction policies for managing limited cache memory."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Cache eviction policies for managing limited cache memory.

## How It Works

**For Beginners:** When running AI models, caching helps avoid loading the same model
multiple times, which speeds things up. However, caches have limited space. An eviction
policy determines which items to remove when the cache becomes full. Think of it like
deciding which apps to close on your phone when memory runs low.

- **LRU (Least Recently Used)**: Removes items that haven't been used in the longest time
- **LFU (Least Frequently Used)**: Removes items that are used the least often
- **FIFO (First In First Out)**: Removes the oldest items first, like a queue
- **Random**: Removes items randomly (simplest but least efficient)

Most applications use LRU as it provides a good balance between performance and simplicity.

## Fields

| Field | Summary |
|:-----|:--------|
| `FIFO` | First In First Out - removes the oldest items first, regardless of usage. |
| `LFU` | Least Frequently Used - removes items that have been accessed the fewest times. |
| `LRU` | Least Recently Used - removes items that haven't been accessed in the longest time. |
| `Random` | Random eviction - removes items randomly when cache is full. |

