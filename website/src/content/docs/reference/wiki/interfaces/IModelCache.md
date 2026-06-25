---
title: "IModelCache<T, TInput, TOutput>"
description: "Defines a caching mechanism for storing and retrieving optimization step data during model training."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a caching mechanism for storing and retrieving optimization step data during model training.

## How It Works

This interface provides methods to store, retrieve, and clear intermediate calculation results
during the training process of machine learning models. Caching these results can significantly
improve performance by avoiding redundant calculations.

**For Beginners:** Think of model caching like saving your progress in a video game.

When training machine learning models, the computer performs many calculations in steps:

- These calculations can be time-consuming and resource-intensive
- By saving (caching) the results of each step, we avoid having to redo calculations
- This makes the training process much faster, especially when:
* You're experimenting with different model settings
* You need to pause and resume training
* You want to analyze intermediate results

For example, imagine you're baking a complex cake that requires multiple stages:

- Instead of starting from scratch each time you make a mistake
- You could save the batter at different stages
- If something goes wrong, you can go back to a saved point rather than starting over

This interface provides the methods needed to implement this "save progress" functionality
for machine learning models.

## Methods

| Method | Summary |
|:-----|:--------|
| `CacheStepData(String,OptimizationStepData<,,>)` | Stores optimization step data in the cache with the specified key. |
| `ClearCache` | Removes all cached optimization step data. |
| `GenerateCacheKey(IFullModel<,,>,OptimizationInputData<,,>)` | Generates a deterministic cache key based on the solution model and input data. |
| `GetCachedStepData(String)` | Retrieves previously cached optimization step data associated with the specified key. |

