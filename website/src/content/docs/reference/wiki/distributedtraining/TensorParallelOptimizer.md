---
title: "TensorParallelOptimizer<T, TInput, TOutput>"
description: "Implements Tensor Parallel optimizer - coordinates updates for tensor-parallel layers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements Tensor Parallel optimizer - coordinates updates for tensor-parallel layers.

## For Beginners

This optimizer works with tensor parallel models where individual layers are split.
Since each process only has part of each layer, we need to carefully coordinate
gradient synchronization. Different layer types require different synchronization
patterns (before or after the computation).

## How It Works

**Strategy Overview:**
Tensor parallel optimizer coordinates optimization for models using tensor parallelism.
Different parts of each layer are distributed across processes, requiring careful
synchronization. For column-parallel layers, an AllReduce is needed after computation.
For row-parallel layers, synchronization happens before computation. The optimizer
ensures proper gradient flow and parameter updates across the tensor-parallel group.

**Use Cases:**

- Works with TensorParallelModel
- Transformer models with large layers
- Very wide models

**Trade-offs:**

- Memory: Excellent for wide layers
- Communication: High - frequent synchronization within layers
- Complexity: Very High - layer-specific patterns
- Best for: Large transformers, fast interconnects

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` |  |
| `Optimize(OptimizationInputData<,,>)` |  |
| `Serialize` |  |
| `SynchronizeOptimizerState` |  |

