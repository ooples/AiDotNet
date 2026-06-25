---
title: "LocalSGDOptimizer<T, TInput, TOutput>"
description: "Implements Local SGD distributed training optimizer - parameter averaging after local optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements Local SGD distributed training optimizer - parameter averaging after local optimization.

## For Beginners

Unlike traditional DDP which synchronizes gradients before every parameter update, Local SGD
lets each worker train independently for several steps, then averages the final model parameters.
Think of it like students studying independently for a week, then meeting to average their
understanding, rather than checking answers after every practice problem.

## How It Works

**Strategy Overview:**
Local SGD allows each worker to perform multiple local optimization steps independently,
then synchronizes model parameters (not gradients) across all workers using AllReduce averaging.
This reduces communication frequency compared to traditional DDP while maintaining convergence.
Based on "Don't Use Large Mini-Batches, Use Local SGD" (Lin et al., 2020).

**Key Difference from DDP:**

- **Local SGD (this class)**: Optimize locally → Average PARAMETERS → Continue training
- **True DDP**: Compute gradients → Average GRADIENTS → Apply averaged gradients → Continue training

**Use Cases:**

- Reducing communication frequency in distributed training
- Slower network connections where communication is expensive
- Works with any optimizer (Adam, SGD, RMSprop, etc.)
- Large models where parameter synchronization dominates training time

**Trade-offs:**

- Memory: Each process stores full model and optimizer state
- Communication: Very low - parameters synchronized less frequently than gradients
- Convergence: Slightly different trajectory than DDP but reaches similar final accuracy
- Complexity: Low - straightforward parameter averaging
- Best for: Communication-constrained distributed training

**Production Note:**
For true DDP (gradient averaging), use GradientCompressionOptimizer with compression ratio = 1.0,
which properly averages gradients before parameter updates.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LocalSGDOptimizer(IOptimizer<,,>,IShardingConfiguration<>)` | Creates a Local SGD optimizer that averages parameters across workers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` |  |
| `Optimize(OptimizationInputData<,,>)` |  |
| `Serialize` |  |
| `SynchronizeOptimizerState` |  |

