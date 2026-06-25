---
title: "ElasticDistributedSampler"
description: "Distributed sampler that evenly divides data across multiple workers with elastic scaling support."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

Distributed sampler that evenly divides data across multiple workers with elastic scaling support.

## How It Works

Ensures each distributed worker sees a non-overlapping subset of the data each epoch.
Supports dynamic resizing when workers join or leave (elastic training).
Each worker gets dataset_size / num_replicas samples per epoch.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ElasticDistributedSampler(ElasticDistributedSamplerOptions)` | Creates a new elastic distributed sampler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DatasetSize` | Gets the total dataset size. |
| `Length` |  |
| `NumReplicas` | Gets the current number of replicas. |
| `Rank` | Gets the current rank. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetIndicesCore` |  |
| `Rescale(Int32,Int32)` | Dynamically updates the number of replicas and rank for elastic scaling. |

