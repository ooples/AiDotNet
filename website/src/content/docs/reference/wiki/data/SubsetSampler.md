---
title: "SubsetSampler"
description: "A sampler that returns a subset of indices."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

A sampler that returns a subset of indices.

## How It Works

SubsetSampler is useful when you want to train on a specific subset of your data,
such as a validation split or a filtered dataset.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SubsetSampler(IEnumerable<Int32>,Boolean,Nullable<Int32>)` | Initializes a new instance of the SubsetSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Length` |  |
| `Shuffle` | Gets or sets whether to shuffle the indices each epoch. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetIndicesCore` |  |

