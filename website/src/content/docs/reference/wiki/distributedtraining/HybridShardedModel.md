---
title: "HybridShardedModel<T, TInput, TOutput>"
description: "Implements 3D Parallelism (Hybrid Sharded) model - combines data, tensor, and pipeline parallelism."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements 3D Parallelism (Hybrid Sharded) model - combines data, tensor, and pipeline parallelism.

## For Beginners

3D Parallelism is the ultimate distributed training strategy - it combines ALL the techniques:

Imagine training a MASSIVE model across 512 GPUs:

- Pipeline Parallel (depth): Split model into 8 stages (64 GPUs per stage)
- Tensor Parallel (width): Within each stage, split layers 8 ways (8 GPUs per tensor group)
- Data Parallel (batches): Remaining 8 GPUs in each tensor group process different data

Layout example for 512 GPUs = 8 pipeline × 8 tensor × 8 data:

- Stage 0: GPUs 0-63 (layers 0-12)
- Tensor group 0: GPUs 0-7 (data replicas)
- Tensor group 1: GPUs 8-15 (data replicas)
- ... 8 tensor groups total
- Stage 1: GPUs 64-127 (layers 13-25)
- ...and so on

## How It Works

**Strategy Overview:**
3D Parallelism combines all three major parallelism strategies for maximum scalability:

- Data Parallelism: Different data batches across replicas
- Tensor Parallelism: Layer-wise partitioning within each pipeline stage
- Pipeline Parallelism: Model depth partitioning across stages

This enables training extremely large models (100B+ parameters) on thousands of GPUs by
exploiting parallelism in all dimensions. This is the strategy used for training models
like GPT-3, Megatron-Turing NLG, and other frontier models.

**Use Cases:**

- Training frontier models (GPT-3 scale: 100B-1T parameters)
- Requires 100s to 1000s of GPUs
- When single parallelism dimension isn't enough
- Production training at largest scales (OpenAI, Google, Meta)

**Trade-offs:**

- Memory: Excellent - exploits all memory-saving strategies
- Communication: Complex - requires careful network topology optimization
- Complexity: Very High - most complex distributed strategy
- Best for: Frontier-scale models (100B+ params), massive GPU clusters
- Requires: Careful tuning of all three parallelism dimensions for efficiency

**Implementation Note:**
This is a production-ready framework providing the 3D parallelism infrastructure.
Full production deployment requires:

1. Process group management (separate groups for data/tensor/pipeline)
2. Model-specific layer partitioning
3. Careful configuration tuning for your specific cluster topology

This implementation demonstrates the pattern and provides the foundation.

Example:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HybridShardedModel(IFullModel<,,>,IShardingConfiguration<>,Int32,Int32,Int32)` | Creates a new 3D Parallel (Hybrid Sharded) model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `InitializeSharding` | Initializes 3D parallelism by partitioning along all dimensions. |
| `LoadModel(String)` |  |
| `OnBeforeInitializeSharding` | Called before InitializeSharding to set up derived class state. |
| `Predict()` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `StoreConfigAndPassThrough(IFullModel<,,>,Int32,Int32,Int32)` | Stores constructor parameters in ThreadLocal before base constructor call. |
| `SynchronizeGradients` |  |
| `Train(,)` |  |
| `WithParameters(Vector<>)` |  |

