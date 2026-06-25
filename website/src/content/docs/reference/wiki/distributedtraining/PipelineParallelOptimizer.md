---
title: "PipelineParallelOptimizer<T, TInput, TOutput>"
description: "Implements Pipeline Parallel optimizer - coordinates optimization across pipeline stages."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements Pipeline Parallel optimizer - coordinates optimization across pipeline stages.

## For Beginners

This optimizer works with pipeline parallel models where the model is split into stages.
It handles the complexity of gradient accumulation - since we process multiple micro-batches
through the pipeline, gradients need to be accumulated before the final parameter update.
Think of it like collecting feedback from multiple practice sessions before making adjustments.

## How It Works

**Strategy Overview:**
Pipeline parallel optimizer coordinates optimization across different pipeline stages.
Each stage optimizes its own layer parameters, with gradient accumulation across micro-batches.
The optimizer ensures proper synchronization between forward and backward passes through the
pipeline, handling the gradient accumulation from multiple micro-batches.

**Use Cases:**

- Works with PipelineParallelModel
- Very deep models split into stages
- Handles micro-batch gradient accumulation

**Trade-offs:**

- Memory: Good for deep models
- Communication: Low between stages
- Complexity: High - gradient accumulation, pipeline scheduling
- Best for: Deep models with pipeline parallelism

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` |  |
| `Optimize(OptimizationInputData<,,>)` |  |
| `Serialize` |  |
| `SynchronizeOptimizerState` |  |

