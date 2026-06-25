---
title: "StratifiedSampler"
description: "A sampler that ensures each class is represented proportionally in each epoch."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

A sampler that ensures each class is represented proportionally in each epoch.

## For Beginners

When your dataset has unequal class sizes (e.g., 90% cats, 10% dogs),
random sampling might sometimes produce batches with only cats. Stratified sampling
ensures every batch has a similar ratio of cats to dogs as the full dataset.

Example:

## How It Works

StratifiedSampler maintains the class distribution from the original dataset
during sampling. This is especially important for imbalanced datasets where
some classes have many more samples than others.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedSampler(IEnumerable<Int32>,Int32,Nullable<Int32>)` | Initializes a new instance of the StratifiedSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Labels` |  |
| `Length` |  |
| `NumClasses` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildClassIndices` | Builds the index mapping for each class. |
| `GetIndicesCore` |  |

