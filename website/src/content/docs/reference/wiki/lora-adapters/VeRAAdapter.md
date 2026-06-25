---
title: "VeRAAdapter"
description: "VeRA (Vector-based Random Matrix Adaptation) adapter - an extreme parameter-efficient variant of LoRA."
section: "Reference"
---

_LoRA / PEFT Adapters_

VeRA (Vector-based Random Matrix Adaptation) adapter - an extreme parameter-efficient variant of LoRA.

## For Beginners

VeRA is an ultra-efficient version of LoRA for extreme memory constraints.

Think of the difference this way:

- Standard LoRA: Each layer has its own pair of small matrices (A and B) that are trained
- VeRA: ALL layers share the same random matrices (A and B) which are frozen. Only tiny

scaling vectors are trained per layer.

Example parameter comparison for a 1000x1000 layer with rank=8:

- Full fine-tuning: 1,000,000 parameters
- Standard LoRA (rank=8): 16,000 parameters (98.4% reduction)
- VeRA (rank=8): ~1,600 parameters (99.84% reduction) - 10x fewer than LoRA!

Trade-offs:

- ✅ Extreme parameter efficiency (10x fewer than LoRA)
- ✅ Very low memory footprint
- ✅ Shared matrices reduce storage when adapting many layers
- ⚠️ Slightly less flexible than standard LoRA (shared random projection)
- ⚠️ Performance may be marginally lower than LoRA in some cases

When to use VeRA:

- Extreme memory constraints (mobile, edge devices)
- Fine-tuning many layers with limited resources
- Rapid prototyping with minimal parameter overhead
- When LoRA is still too expensive

## How It Works

VeRA achieves 10x fewer trainable parameters than standard LoRA by:

- Using a single pair of random low-rank matrices (A and B) shared across ALL layers
- Freezing these shared matrices (they are never trained)
- Training only small scaling vectors (d and b) that are specific to each layer

The forward computation is: output = base_layer(input) + d * (B * A * input) * b
where d and b are trainable vectors, and A and B are frozen shared matrices.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new VeRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured VeRAAdapter (rank {config.Rank}).");
```

