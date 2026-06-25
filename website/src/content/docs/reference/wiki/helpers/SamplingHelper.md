---
title: "SamplingHelper"
description: "Provides methods for sampling data, which is essential for many AI and machine learning techniques."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides methods for sampling data, which is essential for many AI and machine learning techniques.

## How It Works

**For Beginners:** Sampling is like picking random items from a collection. This is important in AI
for creating training sets, validation sets, and implementing techniques like bootstrapping
that help improve model accuracy and reliability.

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentRandom` | Gets the random number generator used for all sampling operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearSeed` | Clears the seed and restores thread-safe random number generation. |
| `CreateBootstrapSamples([],Int32,Nullable<Int32>)` | Creates bootstrap samples from the given data, which are random samples with replacement used for estimating statistical properties. |
| `SampleWithReplacement(Int32,Int32)` | Performs sampling with replacement, meaning the same item can be selected multiple times. |
| `SampleWithoutReplacement(Int32,Int32)` | Performs sampling without replacement, meaning once an item is selected, it cannot be selected again. |
| `SetSeed(Int32)` | Sets the seed for the random number generator to ensure reproducible results. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_seededRandom` | Seeded random instance for reproducible sampling. |

