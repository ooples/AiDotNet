---
title: "ZeRO1Model<T, TInput, TOutput>"
description: "Implements ZeRO Stage 1 model wrapper - shards optimizer states only."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements ZeRO Stage 1 model wrapper - shards optimizer states only.

## For Beginners

This class implements ZeRO Stage 1, which is a middle ground between DDP and full sharding.
The model parameters and gradients are replicated (like DDP), but the optimizer's internal state
(like momentum buffers in Adam) is split across processes to save memory.

## How It Works

**Strategy Overview:**
ZeRO Stage 1 (Zero Redundancy Optimizer) shards only the optimizer states (momentum, variance buffers)
across processes while keeping parameters and gradients replicated. This provides memory savings
for the optimizer state without changing the training communication pattern significantly.

Think of it like a team where everyone has the full playbook (parameters) and shares all their
notes (gradients), but each person keeps their own personal training journal (optimizer state)
rather than everyone keeping a copy of everyone's journals.

**Use Cases:**

- When optimizer state is large (Adam, RMSprop) but model fits in memory
- Want some memory savings without full FSDP complexity
- Gradual migration path from DDP to ZeRO-3/FSDP

**Trade-offs:**

- Memory: Good - saves optimizer state memory (4x for Adam: fp32 params + momentum + variance + gradients)
- Communication: Low - same as DDP (AllReduce gradients)
- Complexity: Low - similar to DDP, works with ZeRO1Optimizer
- Best for: Large models with stateful optimizers, transitioning from DDP

Example:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ZeRO1Model(IFullModel<,,>,IShardingConfiguration<>)` | Creates a new ZeRO-1 model wrapping an existing model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `Deserialize(Byte[])` |  |
| `GetModelMetadata` |  |
| `InitializeSharding` | Initializes ZeRO-1 - no parameter sharding, keeps full parameters like DDP. |
| `LoadModel(String)` |  |
| `Predict()` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SynchronizeGradients` |  |
| `Train(,)` |  |
| `WithParameters(Vector<>)` |  |

