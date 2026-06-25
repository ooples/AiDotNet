---
title: "DeltaLoRAAdapter"
description: "Delta-LoRA adapter that focuses on parameter-efficient delta updates with momentum."
section: "Reference"
---

_LoRA / PEFT Adapters_

Delta-LoRA adapter that focuses on parameter-efficient delta updates with momentum.

## For Beginners

Think of Delta-LoRA as "change-focused" LoRA.

Regular LoRA learns: "What should the weights be?"
Delta-LoRA learns: "How should the weights change?"

This difference matters because:

1. Changes (deltas) often have simpler patterns than absolute values
2. Momentum helps smooth out noisy updates
3. Can converge faster when the optimal adaptation is a smooth transformation

Key concepts:

- **Delta weights**: Accumulated changes to parameters (not the parameters themselves)
- **Delta scaling**: Controls how strongly deltas affect the output
- **Momentum**: Smooths updates by remembering previous changes

When Delta-LoRA works better than standard LoRA:

- Tasks requiring smooth, gradual adaptations
- Fine-tuning where the base model is already close to optimal
- Scenarios with noisy gradients that benefit from momentum
- Transfer learning where you want to preserve more of the original model's behavior

Example: If you're adapting a language model to a new domain, Delta-LoRA can
make smaller, more conservative changes that preserve the model's general knowledge
while adapting to domain-specific patterns.

## How It Works

Delta-LoRA is a variant of LoRA that explicitly models the change (delta) in parameters
rather than the absolute values. This approach can achieve better convergence in certain
scenarios by focusing on the parameter update dynamics with momentum-based accumulation.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new DeltaLoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured DeltaLoRAAdapter (rank {config.Rank}).");
```

