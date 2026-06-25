---
title: "RandomSampler"
description: "A sampler that randomly shuffles the dataset indices each epoch."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

A sampler that randomly shuffles the dataset indices each epoch.

## For Beginners

This sampler shuffles your data randomly before each training epoch.
Shuffling is important because:

- It prevents the model from learning patterns based on data order
- It helps the model generalize better
- It ensures different batches each epoch for varied gradient updates

Example:

## How It Works

RandomSampler is the default sampling strategy for most training scenarios.
It shuffles the dataset indices using the Fisher-Yates algorithm for O(n) time complexity.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomSampler(Int32,Nullable<Int32>)` | Initializes a new instance of the RandomSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Length` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetIndicesCore` |  |

