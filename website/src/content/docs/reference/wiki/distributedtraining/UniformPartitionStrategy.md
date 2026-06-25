---
title: "UniformPartitionStrategy<T>"
description: "Divides model parameters evenly across pipeline stages."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Divides model parameters evenly across pipeline stages.

## For Beginners

This is the default strategy. It splits the model like cutting
a cake into equal slices. It works well when all layers have similar computational cost,
but can cause imbalance when some layers (like attention) are much heavier than others.

## How It Works

This is the simplest partitioning strategy: each stage gets approximately the same
number of parameters. When the total isn't evenly divisible, earlier stages get one
extra parameter each.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputePartition(Int32,Int32)` |  |

