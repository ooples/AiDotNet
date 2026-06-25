---
title: "ZeRO3Model<T, TInput, TOutput>"
description: "Implements ZeRO Stage 3 model wrapper - full sharding of parameters, gradients, and optimizer states."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements ZeRO Stage 3 model wrapper - full sharding of parameters, gradients, and optimizer states.

## For Beginners

ZeRO-3 is identical to FSDP - it's the ultimate memory-saving strategy. Everything is sharded:
parameters, gradients, and optimizer states. Each process only holds a small piece of the model,
and pieces are gathered only when absolutely needed, then immediately released.

## How It Works

**Strategy Overview:**
ZeRO Stage 3 is the full implementation of the ZeRO optimization, sharding parameters, gradients,
AND optimizer states across all processes. This is equivalent to PyTorch's FSDP (Fully Sharded Data Parallel).
Parameters are gathered just-in-time for forward/backward passes and immediately released,
maximizing memory efficiency.

This class is essentially an alias/wrapper for FSDPModel to maintain ZeRO naming consistency.

**Use Cases:**

- Same as FSDP - training very large models
- When you prefer ZeRO terminology over FSDP
- Maximum memory efficiency

**Trade-offs:**

- Same as FSDP
- Memory: Excellent - everything sharded
- Communication: Higher - AllGather for each forward/backward
- Complexity: Moderate

Example:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ZeRO3Model(IFullModel<,,>,IShardingConfiguration<>)` | Creates a new ZeRO-3 model wrapping an existing model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `GetModelMetadata` |  |
| `WithParameters(Vector<>)` |  |

