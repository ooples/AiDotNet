---
title: "DefaultModelCache<T, TInput, TOutput>"
description: "Provides a default implementation of model caching for optimization step data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Caching`

Provides a default implementation of model caching for optimization step data.

## For Beginners

A model cache is like a storage box that keeps track of the progress made during 
machine learning model training.

When training a machine learning model, the system makes many small adjustments (optimization steps) 
to improve the model's accuracy. Each step produces important information about the model's current state.

This cache stores that information for each step, allowing the training process to:

- Resume training from where it left off if interrupted
- Avoid repeating calculations that were already done
- Keep track of how the model is improving over time

Think of it like saving your progress in a video game, so you don't have to start from the beginning 
if you need to take a break.

## Methods

| Method | Summary |
|:-----|:--------|
| `CacheStepData(String,OptimizationStepData<,,>)` | Stores optimization step data in the cache with the specified key. |
| `ClearCache` | Removes all cached optimization step data from the cache. |
| `GenerateCacheKey(IFullModel<,,>,OptimizationInputData<,,>)` | Generates a deterministic cache key based on the solution model and input data using SHA-256 hashing. |
| `GetCachedStepData(String)` | Retrieves cached optimization step data using the specified key. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cache` | The internal dictionary that stores optimization step data, allowing concurrent access from multiple threads. |

