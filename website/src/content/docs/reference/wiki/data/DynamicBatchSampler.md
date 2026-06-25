---
title: "DynamicBatchSampler"
description: "Creates batches that fit a maximum number of tokens/elements rather than a fixed number of samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

Creates batches that fit a maximum number of tokens/elements rather than a fixed number of samples.

## How It Works

Unlike fixed-size batching (N samples per batch), dynamic batching fits as many samples
as possible up to a token/element budget. This maximizes GPU utilization for variable-length
inputs by ensuring each batch uses roughly the same amount of memory regardless of sequence lengths.

For example, with a max tokens budget of 512: a batch might contain 8 sequences of length 64,
or 2 sequences of length 256, or 1 sequence of length 512.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DynamicBatchSampler(Int32[],Int32,Int32,Boolean,Boolean,Nullable<Int32>)` | Creates a new dynamic batch sampler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the maximum number of samples per batch (upper bound even if token budget allows more). |
| `DropLast` |  |
| `Length` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBatchIndices` |  |
| `GetIndicesCore` |  |

