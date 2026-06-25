---
title: "DDPOptimizer<T, TInput, TOutput>"
description: "Implements true DDP (Distributed Data Parallel) optimizer - industry-standard gradient averaging."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements true DDP (Distributed Data Parallel) optimizer - industry-standard gradient averaging.

## For Beginners

DDP works by having each worker compute gradients on their local batch of data, then averaging
those gradients across all workers before updating the model. It's like a study group where everyone
works on different practice problems, shares their solutions, averages the feedback, and everyone
applies the same averaged correction to their understanding.

## How It Works

**Strategy Overview:**
True DDP is the industry-standard distributed training approach used by PyTorch, TensorFlow, and JAX.
After computing gradients on local data, gradients are averaged across all workers using AllReduce,
then the averaged gradients are applied to update model parameters. This ensures all workers
stay perfectly synchronized with identical parameter updates at every step.

**Key Difference from Local SGD:**

- **True DDP (this class)**: Compute gradients → Average GRADIENTS → Apply averaged gradients
- **Local SGD**: Optimize locally → Average PARAMETERS after multiple steps

DDP maintains tighter synchronization but requires more frequent communication.

**How It Works:**

1. Each worker computes gradients on local data batch
2. Gradients are synchronized via AllReduce (averaging across all workers)
3. Each worker applies the same averaged gradients to their model
4. All workers now have identical parameters
5. Repeat for next iteration

**Use Cases:**

- Standard multi-GPU distributed training (PyTorch DDP, TensorFlow MirroredStrategy)
- Fast interconnects (NVLink, InfiniBand) where communication is cheap
- Training where tight synchronization is critical
- Works with any optimizer (SGD, Adam, RMSprop, etc.)
- Default choice for distributed training with good network

**Trade-offs:**

- Memory: Each process stores full model and optimizer state
- Communication: Moderate - gradients synchronized every step (can use gradient compression)
- Synchronization: Perfect - all workers always have identical parameters
- Convergence: Identical to single-GPU training (mathematically equivalent)
- Complexity: Low - straightforward gradient averaging
- Best for: Fast networks, standard distributed training scenarios

**Production Implementation:**
This implementation uses the gradient access infrastructure (LastComputedGradients, ApplyGradients)
to properly average gradients before parameter updates. It reverses local gradient applications
to recover original parameters, applies averaged gradients, ensuring true DDP semantics.

**Industry Standard:**
This implementation matches PyTorch's DistributedDataParallel, TensorFlow's MirroredStrategy,
and JAX's pmap with gradient averaging. It is the gold standard for distributed training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DDPOptimizer(IOptimizer<,,>,IShardingConfiguration<>)` | Creates a true DDP optimizer that averages gradients across workers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` |  |
| `Optimize(OptimizationInputData<,,>)` |  |
| `Serialize` |  |
| `SynchronizeOptimizerState` |  |

