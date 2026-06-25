---
title: "ZeRO1Optimizer<T, TInput, TOutput>"
description: "Implements ZeRO Stage 1 optimizer - shards optimizer states only."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements ZeRO Stage 1 optimizer - shards optimizer states only.

## For Beginners

This optimizer saves memory by splitting the optimizer's internal memory (like momentum in Adam)
across processes. The model parameters are still fully replicated, but each process only stores
a portion of the optimizer's "memory" or "state". When it's time to update parameters, processes
share their pieces of the optimizer state as needed.

## How It Works

**Strategy Overview:**
ZeRO-1 optimizer shards optimizer states (momentum buffers, variance estimates) across processes
while keeping parameters and gradients replicated. This reduces memory overhead from optimizer
state (which can be 4x the model size for Adam: fp32 params + momentum + variance + gradients).
When needed for updates, optimizer states are gathered from their respective owners.

**Use Cases:**

- Using stateful optimizers (Adam, RMSprop) with limited memory
- Want memory savings without full ZeRO-3/FSDP complexity
- Works well with ZeRO1Model

**Trade-offs:**

- Memory: Good - saves ~4x memory from optimizer states for Adam
- Communication: Low - same as DDP plus occasional state gather
- Complexity: Moderate - state sharding adds some complexity
- Best for: Memory-constrained scenarios with stateful optimizers

## Methods

| Method | Summary |
|:-----|:--------|
| `Deserialize(Byte[])` |  |
| `Optimize(OptimizationInputData<,,>)` |  |
| `Serialize` |  |
| `SynchronizeOptimizerState` |  |

