---
title: "LoRAFAAdapter"
description: "LoRA-FA (LoRA with Frozen A matrix) adapter for parameter-efficient fine-tuning."
section: "Reference"
---

_LoRA / PEFT Adapters_

LoRA-FA (LoRA with Frozen A matrix) adapter for parameter-efficient fine-tuning.

## For Beginners

LoRA-FA makes LoRA even more efficient!

Standard LoRA uses two small matrices (A and B) that both get trained:

- Matrix A: Compresses input (trained)
- Matrix B: Expands to output (trained)

LoRA-FA optimizes this further:

- Matrix A: Compresses input (frozen - never changes after initialization)
- Matrix B: Expands to output (trained - the only thing that learns)

Why freeze matrix A?

- Research shows matrix A can be randomly initialized and frozen without much performance loss
- This cuts trainable parameters in half (only matrix B is trained)
- Training is faster and uses less memory
- Perfect when you need maximum efficiency

Example parameter counts for a 1000×1000 layer with rank=8:

- Standard LoRA: 8,000 (A) + 8,000 (B) = 16,000 trainable parameters
- LoRA-FA: 0 (A frozen) + 8,000 (B) = 8,000 trainable parameters (50% reduction!)

When to use LoRA-FA:

- Memory is very limited
- Training speed is critical
- You can tolerate a small performance trade-off
- You're working with very large models

## How It Works

LoRA-FA is a variant of standard LoRA that freezes matrix A after random initialization and only
trains matrix B. This provides approximately 50% parameter reduction compared to standard LoRA
with minimal performance loss in most scenarios.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new LoRAFAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured LoRAFAAdapter (rank {config.Rank}).");
```

