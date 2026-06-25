---
title: "HybridShardedOptimizer<T, TInput, TOutput>"
description: "Implements 3D Parallelism optimizer - coordinates across data, tensor, and pipeline dimensions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements 3D Parallelism optimizer - coordinates across data, tensor, and pipeline dimensions.

## For Beginners

This is the most complex optimizer, coordinating all three types of parallelism.
It needs to handle:

1. Averaging gradients across data-parallel replicas (GPUs processing different batches)
2. Synchronizing tensor-parallel groups (GPUs sharing layer computations)
3. Accumulating gradients from pipeline micro-batches

Think of it like coordinating a massive team split into departments (pipeline stages),
work groups (tensor parallel), and shifts (data parallel) - all need to sync at the right times.

## How It Works

**Strategy Overview:**
3D Parallelism optimizer coordinates optimization across all three parallelism dimensions:

- Data parallel: synchronizes gradients across data-parallel replicas
- Tensor parallel: synchronizes within tensor-parallel groups
- Pipeline parallel: handles gradient accumulation across micro-batches

This requires managing separate communication groups for each dimension and ensuring
proper synchronization order to maintain correctness and efficiency.

**Use Cases:**

- Frontier-scale models (100B+ parameters)
- 100s to 1000s of GPUs
- Works with HybridShardedModel

**Trade-offs:**

- Memory: Excellent - exploits all dimensions
- Communication: Complex - multiple sync patterns
- Complexity: Very High - most complex optimizer
- Best for: Largest scale training

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` |  |
| `Optimize(OptimizationInputData<,,>)` |  |
| `PerformReduction(List<Vector<>>,ReductionOperation)` | Performs the actual reduction operation on a list of vectors. |
| `Serialize` |  |
| `SubgroupAllReduce(Vector<>,List<Int32>,ReductionOperation)` | Performs AllReduce within a subgroup of ranks. |
| `SubgroupAllReduceGlobal(Vector<>,List<Int32>,ReductionOperation,Boolean)` | Implements subgroup AllReduce using global collective operations (fallback for NCCL, Gloo). |
| `SubgroupAllReduceP2P(Vector<>,List<Int32>,ReductionOperation,Boolean)` | Implements subgroup AllReduce using point-to-point Send/Receive (efficient). |
| `SynchronizeOptimizerState` |  |

