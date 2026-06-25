---
title: "LoadBalancedPartitionStrategy<T>"
description: "Partitions model parameters across pipeline stages using estimated computational cost per layer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Partitions model parameters across pipeline stages using estimated computational cost per layer.

## For Beginners

Imagine an assembly line where some tasks take much longer than others.
If you assign tasks purely by count, some workers finish early and wait while others are still busy.
This strategy assigns tasks by estimated time, so all workers finish at roughly the same time.

For neural networks, attention layers are much more expensive than simple normalization layers,
so this strategy gives fewer attention layers to each stage to balance the workload.

The cost function estimates FLOPs (floating point operations) for a block of parameters:

- Dense/linear layers: ~2 * inputSize * outputSize FLOPs
- Attention: ~4 * seqLen * d_model FLOPs
- LayerNorm: ~5 * d_model FLOPs

Since we don't have layer-level metadata in the parameter vector, costs are estimated from
parameter counts using the heuristic that computation scales quadratically with matrix dimensions.

## How It Works

Instead of dividing parameters uniformly, this strategy uses a cost function to estimate
the computational load for each parameter group (layer). It then assigns parameters to stages
so that each stage has roughly equal total cost, reducing pipeline bubble overhead.

**Reference:** Megatron-LM layer assignment algorithm, NVIDIA 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LoadBalancedPartitionStrategy(Int32,Func<Int32,>)` | Creates a load-balanced partition strategy that auto-detects layer boundaries using a fixed layer size estimate. |
| `LoadBalancedPartitionStrategy(Int32[],Func<Int32,>)` | Creates a load-balanced partition strategy with explicit layer boundaries and optional cost estimator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BacktrackSplitPoints(Matrix<Int32>,Int32,Int32)` | Backtracks from the DP split points to find the optimal layer-to-stage assignment. |
| `ComputePartition(Int32,Int32)` |  |
| `ComputePrefixSums(Int32[],Vector<>)` | Computes prefix sums for parameter sizes and layer costs. |
| `ConvertToPartitions(Int64[],Int32[],Int32)` | Converts layer assignments to parameter partitions (StartIndex, Size). |
| `InitializeDPTables(Vector<>,Int32,Int32)` | Creates and initializes DP tables with base cases for single-stage partitioning. |
| `OptimalPartition(Int32[],Vector<>,Int32)` | Uses dynamic programming to find the partition of layers into stages that minimizes the maximum stage cost (min-max partitioning). |
| `SolveMinMaxDP(Vector<>,Int32,Int32)` | Fills the DP table: dp[s][l] = min of max stage cost assigning layers 0..l-1 to stages 0..s-1. |

