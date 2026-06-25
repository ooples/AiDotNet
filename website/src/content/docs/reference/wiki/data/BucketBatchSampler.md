---
title: "BucketBatchSampler"
description: "Groups sequences by length into buckets, then batches within each bucket to minimize padding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

Groups sequences by length into buckets, then batches within each bucket to minimize padding.

## How It Works

Bucket batching is essential for efficient NLP training. By grouping sequences of similar
lengths together, padding waste is minimized and GPU utilization is maximized.

The sampler sorts samples by their length, divides them into buckets of roughly equal size,
shuffles within each bucket, and yields batches from each bucket.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BucketBatchSampler(Int32[],Int32,Int32,Boolean,Nullable<Int32>)` | Creates a new bucket batch sampler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` |  |
| `DropLast` |  |
| `Length` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBatchIndices` |  |
| `GetIndicesCore` |  |

