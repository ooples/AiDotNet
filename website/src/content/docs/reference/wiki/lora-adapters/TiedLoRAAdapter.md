---
title: "TiedLoRAAdapter"
description: "Tied-LoRA adapter - LoRA with weight tying for extreme parameter efficiency across deep networks."
section: "Reference"
---

_LoRA / PEFT Adapters_

Tied-LoRA adapter - LoRA with weight tying for extreme parameter efficiency across deep networks.

## For Beginners

Tied-LoRA is an ultra-efficient variant of LoRA for deep networks.

Think of the difference this way:

- Standard LoRA: Each layer has its own pair of small matrices (A and B) that are trained
- VeRA: ALL layers share the same random matrices (A and B) which are frozen. Only tiny

scaling vectors are trained per layer.

- Tied-LoRA: ALL layers share the same matrices (A and B) which ARE trained. Only a single

scaling factor is trained per layer.

Example parameter comparison for 10 layers of 1000x1000 with rank=8:

- Full fine-tuning: 10,000,000 parameters
- Standard LoRA (rank=8): 160,000 parameters (10 layers × 16,000 params each)
- Tied-LoRA (rank=8): ~16,010 parameters (shared 16,000 + 10 scaling factors)

Benefits of Tied-LoRA:

- ✅ Extreme parameter efficiency for deep networks (scales with depth)
- ✅ Shared matrices enforce consistency across layers
- ✅ Still trainable (unlike VeRA's frozen matrices)
- ✅ Very low memory footprint
- ✅ Faster training (fewer parameters to update)

Trade-offs:

- ⚠️ Less flexible than standard LoRA (shared adaptation across layers)
- ⚠️ Assumes layers benefit from similar adaptations
- ⚠️ May underperform standard LoRA on heterogeneous architectures

When to use Tied-LoRA:

- Very deep networks (transformers with many similar layers)
- Extreme memory constraints
- When layers have similar structure and function
- Rapid prototyping with minimal parameter overhead
- Fine-tuning massive models (GPT, BERT-style architectures)

Research insight: Tied-LoRA works well because in deep networks, many layers learn similar
transformations. By sharing the LoRA matrices and only varying the strength per layer,
we capture most of the adaptation capability with minimal parameters.

## How It Works

Tied-LoRA achieves even greater parameter efficiency than standard LoRA by:

- Sharing the same LoRA matrices (A and B) across multiple layers
- Training only layer-specific scaling factors
- Particularly effective for very deep networks with many similar layers

The forward computation is: output = base_layer(input) + layerScaling * (B_shared * A_shared * input)
where layerScaling is a trainable scalar unique to each layer, and A and B are shared trainable matrices.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new TiedLoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured TiedLoRAAdapter (rank {config.Rank}).");
```

