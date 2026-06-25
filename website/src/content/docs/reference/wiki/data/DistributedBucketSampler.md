---
title: "DistributedBucketSampler"
description: "Combines distributed partitioning with bucket batching for efficient distributed NLP training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

Combines distributed partitioning with bucket batching for efficient distributed NLP training.

## How It Works

Each rank receives a disjoint partition of the dataset (like DistributedSampler), then
within that partition, samples are grouped by length into buckets for efficient batching
(like BucketBatchSampler). This minimizes padding waste in distributed sequence model training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistributedBucketSampler(Int32[],Int32,Int32,Int32,Int32,Boolean,Boolean,Nullable<Int32>)` | Creates a new distributed bucket sampler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropLast` |  |
| `Length` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBatchIndices` |  |
| `GetIndicesCore` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_batchSize` |  |

