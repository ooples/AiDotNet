---
title: "TrainingMemoryManager<T>"
description: "Manages memory optimization during neural network training including gradient checkpointing, activation pooling, and model sharding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Training.Memory`

Manages memory optimization during neural network training including gradient checkpointing,
activation pooling, and model sharding.

## For Beginners

This manager helps you train larger neural networks by:

1. **Gradient Checkpointing**: Saves memory by recomputing activations during backward pass
2. **Activation Pooling**: Reuses tensor memory to reduce garbage collection
3. **Model Sharding**: Distributes layers across multiple GPUs

Example usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrainingMemoryManager(TrainingMemoryConfig,IEnumerable<ILayer<>>)` | Initializes a new instance of the TrainingMemoryManager. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` | Gets the memory configuration. |
| `IsCheckpointingEnabled` | Gets whether gradient checkpointing is enabled. |
| `IsPoolingEnabled` | Gets whether activation pooling is enabled. |
| `IsShardingEnabled` | Gets whether model sharding is enabled. |
| `PoolStats` | Gets pool statistics if activation pooling is enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackwardSequence(IReadOnlyList<ILayer<>>,Tensor<>)` | Performs backward pass through multiple layers with recomputation. |
| `ClearCheckpoints` | Clears all stored checkpoints to free memory. |
| `ComputeCheckpointIndices(ILayeredModel<>)` | Determines which layers should be checkpointed using layer-aware metadata from an `ILayeredModel`. |
| `ComputeCheckpointIndices(Int32,IReadOnlyList<String>)` | Determines which layers should be checkpointed based on configuration. |
| `Dispose` | Disposes resources used by the memory manager. |
| `EstimateMemorySavings(Int64,Int32,Int32)` | Estimates memory savings from current configuration. |
| `ForwardSequence(IEnumerable<ILayer<>>,Tensor<>)` | Performs forward pass through multiple layers with checkpointing. |
| `ForwardWithCheckpoint(ILayer<>,Tensor<>,Int32)` | Performs a forward pass with checkpointing for a single layer. |
| `GetPoolMemoryUsage` | Gets current memory usage from the activation pool. |
| `RentTensor(Int32[])` | Rents a tensor from the activation pool. |
| `ReturnTensor(Tensor<>)` | Returns a tensor to the activation pool. |
| `ShardedForward(Tensor<>)` | Performs forward pass through sharded model. |
| `ShouldCheckpoint(Int32)` | Determines if a specific layer should be checkpointed. |

