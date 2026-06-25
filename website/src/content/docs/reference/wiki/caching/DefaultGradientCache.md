---
title: "DefaultGradientCache<T>"
description: "Provides a default implementation of the gradient caching mechanism for symbolic models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Caching`

Provides a default implementation of the gradient caching mechanism for symbolic models.

## For Beginners

A gradient cache is like a memory bank that stores pre-calculated mathematical operations
that are frequently used during machine learning model training.

In machine learning, "gradients" are calculations that show how much a model's error would change
if we slightly adjusted a specific parameter. These calculations can be complex and time-consuming,
especially if they need to be repeated many times.

By storing these calculations in a cache (a temporary storage area), we can avoid recalculating the
same gradients repeatedly, which makes the training process much faster. Think of it like remembering
the answer to a difficult math problem so you don't have to solve it again when you need the same answer later.

This class provides a simple way to store and retrieve these gradient calculations using string keys
(like names or identifiers) to look them up quickly.

## Methods

| Method | Summary |
|:-----|:--------|
| `CacheGradient(String,IGradientModel<>)` | Stores a gradient model in the cache with the specified key. |
| `ClearCache` | Removes all cached gradient models from the cache. |
| `GetCachedGradient(String)` | Retrieves a cached gradient model using the specified key. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cache` | The internal dictionary that stores gradient models, allowing concurrent access from multiple threads. |

