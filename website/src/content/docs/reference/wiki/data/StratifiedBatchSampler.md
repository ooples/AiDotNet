---
title: "StratifiedBatchSampler"
description: "A batch sampler that ensures each batch contains samples from all classes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

A batch sampler that ensures each batch contains samples from all classes.

## For Beginners

While StratifiedSampler ensures the overall epoch
has the right class balance, StratifiedBatchSampler ensures EACH BATCH
has balanced classes. This is helpful when:

- Using batch normalization (needs balanced statistics per batch)
- Doing contrastive learning (needs diverse samples in each batch)

## How It Works

StratifiedBatchSampler creates batches where each batch has approximately
the same class distribution. This is useful for batch normalization layers
or when batch-level statistics are important.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedBatchSampler(IEnumerable<Int32>,Int32,Int32,Boolean,Nullable<Int32>)` | Initializes a new instance of the StratifiedBatchSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` |  |
| `DropLast` |  |
| `Labels` |  |
| `Length` |  |
| `NumClasses` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetBatchIndices` |  |
| `GetIndicesCore` |  |

