---
title: "ASHAOptimizer<T, TInput, TOutput>"
description: "Implements ASHA (Asynchronous Successive Halving Algorithm) for hyperparameter optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.HyperparameterOptimization`

Implements ASHA (Asynchronous Successive Halving Algorithm) for hyperparameter optimization.

## How It Works

**For Beginners:** ASHA is an improved version of Hyperband that:

- Doesn't wait for all configurations at a level before promoting some
- Can promote promising configurations as soon as they outperform enough peers
- Is naturally suited for parallel/distributed training
- Typically converges faster than synchronous Hyperband

Key concepts:

- Rungs: Resource levels at which we evaluate (e.g., epochs 1, 3, 9, 27)
- Promotion: Moving a configuration to train with more resources
- Early Stopping: Killing configurations that aren't competitive

ASHA uses the same exponential resource increase as Hyperband but allows
asynchronous promotion based on relative performance.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ASHAOptimizer(Boolean,Int32,Int32,Int32,Double,Nullable<Int32>)` | Initializes a new instance of the ASHAOptimizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Rungs` | Gets the resource levels (rungs) in this ASHA configuration. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBestConfiguration` | Gets the best configuration found at the highest rung. |
| `GetRungStatistics` | Gets statistics about configurations at each rung. |
| `Optimize(Func<Dictionary<String,Object>,>,HyperparameterSearchSpace,Int32)` | Searches for the best hyperparameter configuration using ASHA. |
| `SampleRandomConfiguration(HyperparameterSearchSpace)` | Samples a random configuration from the search space. |
| `ShouldPromote(Int32,)` | Determines if a configuration should be promoted to the next rung. |
| `SuggestNext(HyperparameterTrial<>)` | Suggests the next hyperparameter configuration to try. |

