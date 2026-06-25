---
title: "DiffusionMemoryManager<T>"
description: "Memory management utilities for diffusion models including gradient checkpointing, activation pooling, and model sharding integration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Memory`

Memory management utilities for diffusion models including gradient checkpointing,
activation pooling, and model sharding integration.

## For Beginners

Training large models uses a lot of memory because we need to store:

1. The model parameters
2. Intermediate activations (outputs from each layer during forward pass)
3. Gradients for each parameter

This class helps reduce memory usage through several techniques:

**Gradient Checkpointing:**

- Instead of storing all activations, only store "checkpoints"
- During backward pass, recompute activations between checkpoints
- Trades ~30% more compute time for ~50% less memory

**Activation Pooling:**

- Reuse tensor memory instead of allocating new tensors
- Reduces GC pressure and memory fragmentation

**Model Sharding:**

- Split large models across multiple GPUs
- Each GPU only holds part of the model

## How It Works

This class provides memory-efficient training utilities specifically designed for
large diffusion models (UNet, VAE, etc.) that may not fit in GPU memory during training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionMemoryManager(DiffusionMemoryConfig,IEnumerable<ILayer<>>)` | Initializes a new instance of the DiffusionMemoryManager class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointingEnabled` | Whether gradient checkpointing is enabled. |
| `Config` | Memory configuration. |
| `PoolingEnabled` | Whether activation pooling is enabled. |
| `ShardingEnabled` | Whether model sharding is active. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Checkpoint(Func<ComputationNode<>>,IEnumerable<ComputationNode<>>)` | Wraps a function with gradient checkpointing for memory-efficient training. |
| `CheckpointSequence(IReadOnlyList<Func<ComputationNode<>,ComputationNode<>>>,ComputationNode<>)` | Applies checkpointing to a sequence of layer functions. |
| `EstimateMemory(Int32,Int64)` | Estimates memory savings from current configuration. |
| `ForwardWithCheckpointing(IReadOnlyList<ILayer<>>,Tensor<>)` | Executes a forward pass through layers with optional checkpointing. |
| `GetDeviceMemoryUsage` | Gets memory usage per device. |
| `GetPoolStats` | Gets pooling statistics if available. |
| `RentTensor(Int32[])` | Rents a tensor from the activation pool. |
| `ReturnTensor(Tensor<>)` | Returns a tensor to the activation pool for reuse. |
| `ShardedForward(Tensor<>)` | Performs forward pass through sharded model. |
| `ShardedForward(Tensor<>,Tensor<>)` | Performs forward pass through sharded model with context. |
| `ShardedUpdateParameters()` | Updates parameters across all shards. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |
| `_activationPool` | Activation pool for tensor reuse. |
| `_modelShard` | Model sharding configuration (if multi-GPU). |

