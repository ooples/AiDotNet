---
title: "SLoRAAdapter"
description: "S-LoRA adapter for scalable serving of thousands of concurrent LoRA adapters."
section: "Reference"
---

_LoRA / PEFT Adapters_

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

