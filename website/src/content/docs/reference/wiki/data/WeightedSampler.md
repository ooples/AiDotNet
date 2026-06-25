---
title: "WeightedSampler<T>"
description: "A sampler that samples indices based on their weights."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

A sampler that samples indices based on their weights.

## For Beginners

This sampler lets you control how often each sample appears:

- Higher weight = more likely to be selected
- Lower weight = less likely to be selected

Common uses:

- **Class imbalance**: Give higher weights to underrepresented classes
- **Hard example mining**: Give higher weights to samples the model struggles with
- **Curriculum learning**: Adjust weights during training based on difficulty

Example:

## How It Works

WeightedSampler selects samples with probability proportional to their weights.
This is useful for handling class imbalance, importance sampling, or focusing
training on harder examples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeightedSampler(IEnumerable<>,Nullable<Int32>,Boolean,Nullable<Int32>)` | Initializes a new instance of the WeightedSampler class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateBalancedWeights(IReadOnlyList<Int32>,Int32)` | Creates weights that balance class frequencies. |
| `GetIndicesCore` |  |

