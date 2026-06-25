---
title: "DistributedSampler"
description: "Partitions dataset indices across N ranks for distributed (multi-GPU/multi-node) training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

Partitions dataset indices across N ranks for distributed (multi-GPU/multi-node) training.

## How It Works

Each rank sees a disjoint 1/N subset of the data. Optionally pads the dataset so all ranks
receive exactly the same number of samples (required when using AllReduce-style gradient
synchronization). Equivalent to PyTorch's DistributedSampler.

At each epoch, the sampler shuffles the full dataset deterministically using the epoch number
as the seed modifier, ensuring all ranks agree on the same shuffle order.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistributedSampler(Int32,Int32,Int32,Boolean,Boolean,Nullable<Int32>)` | Creates a new distributed sampler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Length` |  |
| `NumSamplesPerRank` | Number of samples assigned to this rank. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetIndicesCore` |  |

