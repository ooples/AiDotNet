---
title: "SLoRAAdapter<T>"
description: "S-LoRA adapter for scalable serving of thousands of concurrent LoRA adapters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

S-LoRA adapter for scalable serving of thousands of concurrent LoRA adapters.

## For Beginners

S-LoRA solves a real-world problem in production AI systems.

The problem:

- You have a large base model (like GPT or LLaMA)
- You want to serve thousands of different LoRA adapters (one per customer, task, or use case)
- Each adapter is small (few MB), but thousands of them won't fit in GPU memory
- Naive approaches either: load one adapter at a time (slow) or reserve memory for all (wasteful)

S-LoRA's solution:

- Unified memory pool: Dynamically manage adapter weights and cache together
- Batched computation: Process multiple adapters in parallel efficiently
- Adapter clustering: Group adapters by rank for optimized computation
- On-demand loading: Fetch adapters from CPU to GPU memory only when needed

Key features implemented:

1. **Unified Memory Pool**: Single pool for adapter weights (no pre-allocation waste)
2. **Adapter Clustering**: Group adapters by rank for batched computation
3. **Dynamic Loading**: Load adapters on-demand, evict when not needed
4. **Batched Forward Pass**: Process multiple requests with different adapters simultaneously
5. **Memory Efficiency**: Serve 100x more adapters than naive approaches

Research Paper Reference:
"S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
Ying Sheng, Shiyi Cao, et al. (November 2023)
arXiv:2311.03285

Performance (from paper):

- Throughput: 4x improvement over vLLM, 30x over HuggingFace PEFT
- Adapter capacity: 2,000+ concurrent adapters on single server
- Memory efficiency: 75-90% GPU memory utilization
- Scalability: Superlinear throughput scaling with more GPUs

Example usage:
```cs
// Create S-LoRA serving system for base layer
var sloraAdapter = new SLoRAAdapter<double>(baseLayer, rank: 8);

// Register multiple adapters for different tasks
sloraAdapter.RegisterAdapter("customer_1", adapter1);
sloraAdapter.RegisterAdapter("customer_2", adapter2);
sloraAdapter.RegisterAdapter("task_classification", adapter3);

// Process batched requests efficiently
var outputs = sloraAdapter.BatchForward(inputs, adapterIds);
```

When to use S-LoRA:

- Serving multiple LoRA adapters in production
- Multi-tenant AI systems (one adapter per tenant)
- Task-specific fine-tuning at scale
- Limited GPU memory but many adapters
- Need high throughput with many concurrent users

Differences from standard LoRA:

- Standard LoRA: Single adapter, simple forward/backward pass
- S-LoRA: Multiple adapters, optimized for concurrent serving, memory pooling

## How It Works

S-LoRA (Scalable LoRA) is a system designed for efficient serving of many LoRA adapters simultaneously.
Published in November 2023, it addresses the challenge of deploying thousands of task-specific LoRA adapters
in production environments with limited GPU memory.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new SLoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured SLoRAAdapter (rank {config.Rank}).");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SLoRAAdapter(ILayer<>,Int32,Double,Int32,Boolean)` | Initializes a new S-LoRA adapter for scalable multi-adapter serving. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LoadedAdapterCount` | Gets the number of adapters currently loaded in memory. |
| `MaxLoadedAdapters` | Gets the maximum number of adapters that can be loaded simultaneously. |
| `RankClusterCount` | Gets the number of rank clusters for batched computation optimization. |
| `TotalAdapterCount` | Gets the total number of registered adapters in the pool. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BatchForward(Tensor<>[],String[])` | Performs batched forward pass with multiple adapters simultaneously. |
| `ClearAdapters` | Clears all adapters from the pool (useful for testing or reset). |
| `EvictLRUAdapter` | Evicts the least recently used adapter from the loaded cache. |
| `Forward(Tensor<>,String)` | Performs batched forward pass with a specific adapter. |
| `GetRankCluster(Int32)` | Gets the list of adapter IDs in a specific rank cluster. |
| `GetStatistics` | Gets statistics about the current state of the S-LoRA system. |
| `LoadAdapter(String)` | Loads an adapter from the pool into active memory (simulates GPU loading). |
| `MergeToOriginalLayer` | Merges the primary adapter into the base layer and returns the merged layer. |
| `RegisterAdapter(String,LoRALayer<>,Int32)` | Registers a new adapter in the unified memory pool. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adapterPool` | Unified memory pool storing all registered adapters. |
| `_loadedAdapters` | Adapters currently loaded in "GPU memory" (in-memory cache). |
| `_maxLoadedAdapters` | Maximum number of adapters that can be loaded simultaneously (simulates GPU memory limit). |
| `_rankClusters` | Adapters clustered by rank for efficient batched computation. |
| `_timestamp` | Current timestamp for LRU eviction policy. |

