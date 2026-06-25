---
title: "ZeRO3Optimizer<T, TInput, TOutput>"
description: "Implements ZeRO Stage 3 optimizer - full sharding equivalent to FSDP."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Implements ZeRO Stage 3 optimizer - full sharding equivalent to FSDP.

## For Beginners

ZeRO-3 and FSDP optimizers are the same thing. Use whichever name you prefer.
Everything is sharded for maximum memory efficiency.

## How It Works

**Strategy Overview:**
ZeRO-3 is equivalent to FSDP optimizer - full sharding of parameters, gradients, and optimizer
states. This class is an alias to FSDPOptimizer for ZeRO terminology consistency.

