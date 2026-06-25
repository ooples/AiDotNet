---
title: "HRAAdapter"
description: "HRA (Hybrid Rank Adaptation) adapter that combines low-rank and full-rank updates for optimal parameter efficiency."
section: "Reference"
---

_LoRA / PEFT Adapters_

HRA (Hybrid Rank Adaptation) adapter that combines low-rank and full-rank updates for optimal parameter efficiency.

## For Beginners

HRA is like having two tools instead of one: Standard LoRA problem: - Uses only low-rank updates (compressed, efficient) - Some parameters need precise full-rank updates - Full fine-tuning is too expensive - Need something in between HRA solution: - Most parameters use low-rank updates (efficient, covers 95% of needs) - Critical parameters get full-rank updates (precise, covers remaining 5%) - Automatically learns which parameters are critical - Best quality with minimal parameter overhead Analogy: Think of home renovation: - Low-rank updates: Paint the walls (cheap, covers large area, good enough) - Full-rank updates: Replace key structural beams (expensive, small area, critical) - HRA: Do both where appropriate for best results How it works: 1. Start with LoRA-style low-rank matrices (B * A) 2. Add sparse full-rank updates for most important parameters 3. Track importance scores during training 4. Allocate parameter budget optimally between low-rank and sparse full-rank Benefits: - Better quality than pure LoRA (full-rank updates where needed) - More efficient than full fine-tuning (most updates are low-rank) - Adaptive: learns which parameters need full-rank updates - Flexible: adjustable sparsity budget for full-rank component Use cases: - Tasks where LoRA quality is not quite sufficient - Fine-tuning with specific architectural bottlenecks - When you have slightly more parameter budget than LoRA but much less than full fine-tuning - Domains where certain parameters are known to be critical Example parameter comparison for a 1000x1000 layer: - Full fine-tuning: 1,000,000 parameters - Standard LoRA (rank=8): 16,000 parameters (98.4% reduction) - HRA (rank=8, 1% sparsity): 26,000 parameters (97.4% reduction, better quality) Reference: Based on "Hybrid Rank Adaptation" research combining low-rank and sparse full-rank approaches

## How It Works

HRA addresses a key limitation of standard LoRA: while low-rank updates are efficient, some parameters benefit from full-rank updates. HRA uses a hybrid approach: - Dense low-rank updates for most parameters (efficient, like LoRA) - Sparse full-rank updates for critical parameters (precise, targeted) - Importance-based allocation between the two components 

The forward computation is: output = base_layer(input) + low_rank(input) + sparse_full_rank(input) where the hybrid allocation provides the best of both worlds.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new HRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured HRAAdapter (rank {config.Rank}).");
```

