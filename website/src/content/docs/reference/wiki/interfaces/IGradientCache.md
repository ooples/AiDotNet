---
title: "IGradientCache<T>"
description: "Defines an interface for storing and retrieving pre-computed gradients to improve performance in machine learning models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for storing and retrieving pre-computed gradients to improve performance in machine learning models.

## How It Works

**For Beginners:** This interface defines methods for saving and reusing calculations to make your AI models run faster.

In machine learning, models often need to calculate "gradients" - mathematical directions that show how to 
adjust the model to make better predictions. These calculations can be time-consuming, especially for complex models.

Think of a gradient cache like a notebook where you write down answers to difficult math problems:

- When you solve a problem, you write down the answer in your notebook
- Later, if you need the same answer again, you can just look it up instead of re-solving the problem
- This saves you time and effort

The gradient cache works the same way:

- When a gradient is calculated, it's stored with a unique name (the "key")
- If the same gradient is needed again, it can be retrieved using that name
- This avoids repeating expensive calculations

This is especially useful for:

- Complex models with many parameters
- Models that use the same calculations repeatedly
- Training scenarios where speed is important

## Methods

| Method | Summary |
|:-----|:--------|
| `CacheGradient(String,IGradientModel<>)` | Stores a gradient in the cache with a unique key for later retrieval. |
| `ClearCache` | Removes all cached gradients, freeing up memory. |
| `GetCachedGradient(String)` | Retrieves a previously cached gradient using its unique key. |

