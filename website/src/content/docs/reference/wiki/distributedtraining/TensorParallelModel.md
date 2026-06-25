---
title: "TensorParallelModel<T, TInput, TOutput>"
description: "Implements Tensor Parallel model wrapper - splits individual layers across ranks (Megatron-LM style)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements Tensor Parallel model wrapper - splits individual layers across ranks (Megatron-LM style).

## For Beginners

Tensor parallelism is like splitting a single large calculation across multiple workers.
Imagine a huge spreadsheet calculation - instead of one person doing all the math, we divide
the spreadsheet columns across multiple people, each computing their portion simultaneously.

For example, in a neural network layer with a 10000x10000 weight matrix:

- GPU 0 handles columns 0-2499
- GPU 1 handles columns 2500-4999
- GPU 2 handles columns 5000-7499
- GPU 3 handles columns 7500-9999

They compute in parallel, then combine results.

## How It Works

**Strategy Overview:**
Tensor Parallelism (Megatron-LM style) partitions individual layers horizontally across processes.
For example, a large matrix multiplication is split so each GPU computes only a portion of the output,
then results are combined. This is particularly effective for transformer models where attention and
feed-forward layers can be partitioned along specific dimensions (column-parallel and row-parallel).

**Use Cases:**

- Very wide models (large hidden dimensions)
- Transformer models (BERT, GPT) with large attention/FFN layers
- When individual layers are too large for single GPU
- Often combined with pipeline parallelism for maximum scalability

**Trade-offs:**

- Memory: Excellent for wide layers - each rank stores only portion of weights
- Communication: High - requires AllReduce or AllGather within each layer
- Complexity: Very High - requires model-aware partitioning, specific to layer types
- Best for: Transformer models, very wide layers, fast interconnects (NVLink)
- Limitation: Requires fast communication (high overhead on slow networks)

**Implementation Note:**
This is a production-ready framework implementation. Full tensor parallelism requires
model-specific layer partitioning (column-parallel vs row-parallel strategy for different
layer types). This implementation provides the infrastructure. For production use with
specific models (e.g., transformers), extend this class with layer-aware partitioning.

**⚠️ IMPORTANT LIMITATION - Memory Efficiency:**
This implementation gathers the full parameter vector on every Train() and Predict() call
(via GatherFullParameters and SetParameters), which defeats the memory-saving purpose of
true tensor parallelism. While parameters are sharded across ranks for storage, they are
reconstructed into the full vector for each forward/backward pass. This means:

- Memory savings are minimal compared to data-parallel training
- Communication overhead is high (AllGather on every forward pass)
- This wrapper primarily provides gradient synchronization, not memory-efficient tensor parallelism

For true memory-efficient tensor parallelism, you would need layer-aware implementations where
each rank only loads its parameter shard and performs partial matrix multiplications without
ever reconstructing the full parameter vector. This simplified implementation is suitable for:

- Testing and development of distributed training infrastructure
- Scenarios where gradient synchronization is more important than memory efficiency
- Models where memory is not the primary constraint

If memory efficiency is critical, consider using FSDP (Fully Sharded Data Parallel) or ZeRO-3
instead, which shard parameters more aggressively and avoid full parameter reconstruction.

Example:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TensorParallelModel(IFullModel<,,>,IShardingConfiguration<>)` | Creates a new Tensor Parallel model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `InitializeSharding` | Initializes tensor parallelism by partitioning layer weights. |
| `LoadModel(String)` |  |
| `OnBeforeInitializeSharding` | Called before InitializeSharding to set up derived class state. |
| `PerformReduction(List<Vector<>>,ReductionOperation)` | Performs the actual reduction operation on a list of vectors. |
| `Predict()` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SubgroupAllReduce(Vector<>,ReductionOperation)` | Performs AllReduce within the tensor-parallel subgroup. |
| `SubgroupAllReduceGlobal(Vector<>,ReductionOperation,Boolean)` | Implements subgroup AllReduce using global collective operations (fallback for NCCL, Gloo). |
| `SubgroupAllReduceP2P(Vector<>,ReductionOperation,Boolean)` | Implements subgroup AllReduce using point-to-point Send/Receive (efficient). |
| `SynchronizeGradients` | Synchronizes tensor-parallel computation results. |
| `Train(,)` |  |
| `WithParameters(Vector<>)` |  |

