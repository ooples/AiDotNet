---
title: "DVoRAAdapter"
description: "DVoRA (DoRA + VeRA) adapter - combines DoRA's magnitude-direction decomposition with VeRA's extreme parameter efficiency."
section: "Reference"
---

_LoRA / PEFT Adapters_

DVoRA (DoRA + VeRA) adapter - combines DoRA's magnitude-direction decomposition with VeRA's extreme parameter efficiency.

## For Beginners

DVoRA is the ultimate parameter-efficient adapter.

Think of it as a hybrid technique:

- From DoRA: Separate magnitude (strength) from direction for stability
- From VeRA: Use shared random matrices and tiny scaling vectors for efficiency
- The magic: Apply VeRA's adaptation only to the direction, not the magnitude

Parameter comparison for 1000x1000 layer with rank=8:

- Full fine-tuning: 1,000,000 parameters
- Standard LoRA: 16,000 parameters (98.4% reduction)
- DoRA: 17,000 parameters (LoRA + magnitude vector)
- VeRA: 1,600 parameters (99.84% reduction)
- DVoRA: ~1,600 parameters (same as VeRA!) but with better performance (5.0 vs 4.3)

Benefits:

- ✅ Extremely parameter-efficient (10x fewer than standard LoRA, same as VeRA)
- ✅ Better performance than VeRA alone (5.0 vs 4.3 score)
- ✅ Training stability from DoRA's magnitude-direction decomposition
- ✅ Shared matrices reduce storage when adapting many layers
- ✅ Best choice for extreme memory constraints with quality requirements

Trade-offs:

- ⚠️ Requires shared matrix initialization before use
- ⚠️ Slightly more computation than VeRA (due to normalization)
- ⚠️ More complex than standard adapters (combines two techniques)

When to use DVoRA:

- Extreme memory constraints but need better quality than VeRA
- Mobile/edge deployment with limited resources
- Fine-tuning many layers efficiently
- When you want the absolute best parameter efficiency + quality balance

## How It Works

DVoRA achieves the best of both worlds by:

- Applying DoRA's magnitude-direction decomposition for training stability
- Using VeRA's shared frozen matrices and scaling vectors for extreme parameter efficiency
- Applying the VeRA adaptation only to the direction component (not the magnitude)

**Mathematical Formulation:**
Given pre-trained weights W, DVoRA:

1. Decomposes: W = m * d (magnitude and direction)
2. Applies VeRA to direction: d' = d + d_scale * (B * A * input) * b_scale
3. Normalizes direction: d_norm = d' / ||d'||
4. Recomposes: W' = m * d_norm

Where:

- m: magnitude vector (trainable)
- d: direction matrix (normalized weight vectors)
- A, B: shared frozen random matrices (VeRA style)
- d_scale, b_scale: per-layer trainable scaling vectors (VeRA style)

**Research Context:**
DVoRA scores 5.0 vs VeRA's 4.3 (improvement of 16%) while maintaining ultra-low parameter counts.
It combines DoRA's superior training stability with VeRA's extreme parameter efficiency.

**References:**

- DoRA: "Weight-Decomposed Low-Rank Adaptation" (ICML 2024 Oral)
- VeRA: "Vector-based Random Matrix Adaptation"
- DVoRA: Combines both techniques for optimal efficiency and performance

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new DVoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured DVoRAAdapter (rank {config.Rank}).");
```

