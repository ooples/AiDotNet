---
title: "VBLoRAAdapter"
description: "Vector Bank LoRA (VB-LoRA) adapter that uses shared parameter banks for efficient multi-client deployment."
section: "Reference"
---

_LoRA / PEFT Adapters_

Vector Bank LoRA (VB-LoRA) adapter that uses shared parameter banks for efficient multi-client deployment.

## For Beginners

Think of vector banks like a shared library of building blocks.

Traditional LoRA is like each person having their own complete toolbox:

- Person 1: Full set of tools
- Person 2: Full set of tools
- Person 3: Full set of tools

Result: Lots of duplicate tools

VB-LoRA is like a shared tool library:

- Central tool bank: One of each tool type
- Each person: List of which tools they need (indices)
- Everyone shares the same physical tools

Result: Much fewer tools needed overall

This is especially powerful when many adapters need similar adjustments (common in
multi-task learning or personalization scenarios).

## How It Works

VB-LoRA (2024) introduces vector banks - reusable parameter stores shared across multiple LoRA adapters.
Instead of each adapter having its own complete A and B matrices, VB-LoRA maintains global banks of
column vectors. Each adapter selects which vectors from the banks to use via index arrays.

**Key Innovation:** Vector Bank Architecture

Traditional LoRA:

- Each adapter stores full A (inputSize × rank) and B (rank × outputSize) matrices
- Total parameters for N adapters = N × (inputSize × rank + rank × outputSize)
- No sharing between adapters

VB-LoRA:

- Global BankA contains pooled column vectors (inputSize × bankSize)
- Global BankB contains pooled column vectors (bankSize × outputSize)
- Each adapter stores only indices (which vectors to use from banks)
- Total parameters = (inputSize × bankSize + bankSize × outputSize) + N × 2 × rank × sizeof(int)
- Massive reduction when bankSize << N × rank

**Benefits:**

1. **Reduced Duplication**: Similar adapters share vector bank entries
2. **Lower Communication Overhead**: Multi-client systems can cache banks locally
3. **Memory Efficiency**: Fewer unique parameters to store and transmit
4. **Scalability**: Adding new adapters only requires index arrays, not full matrices
5. **Knowledge Sharing**: Banks capture common adaptation patterns

**Example Scenario:**

Suppose you're deploying personalized language models to 1000 users:

- Traditional LoRA: Each user needs their own 16K parameter adapter (16MB total)
- VB-LoRA: Shared 256K parameter bank + 1000 users × 128 indices each
- Result: 84% memory reduction (256K + 128K vs 16M)

The shared bank captures common language patterns, while per-user indices
select the patterns relevant to each individual.

