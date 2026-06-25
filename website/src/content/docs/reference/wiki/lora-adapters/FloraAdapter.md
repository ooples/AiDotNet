---
title: "FloraAdapter"
description: "Implements Flora (Low-Rank Adapters Are Secretly Gradient Compressors) adapter for memory-efficient fine-tuning."
section: "Reference"
---

_LoRA / PEFT Adapters_

Implements Flora (Low-Rank Adapters Are Secretly Gradient Compressors) adapter for memory-efficient fine-tuning.

## How It Works

Flora reinterprets LoRA as a gradient compression mechanism and achieves high-rank updates through periodic resampling of projection matrices while maintaining sublinear space complexity for optimizer states. 

**Research Paper:** "Flora: Low-Rank Adapters Are Secretly Gradient Compressors" by Yongchang Hao et al., ICML 2024. arXiv:2402.03293 

**Key Innovation:** Unlike standard LoRA which restricts weight updates to a fixed low-rank subspace, Flora periodically resamples the projection matrices (A and B), allowing the effective rank of cumulative updates to grow over time. This achieves performance comparable to full-rank fine-tuning while maintaining the memory efficiency of LoRA.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new FloraAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured FloraAdapter (rank {config.Rank}).");
```

