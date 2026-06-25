---
title: "IPipelinePartitionStrategy<T>"
description: "Defines a strategy for partitioning model parameters across pipeline stages."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a strategy for partitioning model parameters across pipeline stages.

## For Beginners

When splitting a neural network across multiple devices (pipeline parallelism),
you need to decide which layers go on which device. This interface defines that decision.

The default (uniform) strategy just divides parameters evenly, but this can lead to
imbalanced workloads because some layers (like attention) are much more expensive than
others (like layer normalization). A load-balanced strategy can account for this.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePartition(Int32,Int32)` | Computes the partition boundaries for the given number of stages. |

