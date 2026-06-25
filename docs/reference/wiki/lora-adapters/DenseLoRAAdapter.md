---
title: "DenseLoRAAdapter"
description: "LoRA adapter specifically for Dense and FullyConnected layers with 1D input/output shapes."
section: "Reference"
---

_LoRA / PEFT Adapters_

LoRA adapter specifically for Dense and FullyConnected layers with 1D input/output shapes.

## For Beginners

This adapter lets you add LoRA to Dense or FullyConnected layers. Think of it like adding a "correction layer" that learns what adjustments are needed: - The base layer keeps its original weights (optionally frozen) - The LoRA layer learns a small correction - The final output is: original_output + lora_correction This is incredibly useful for fine-tuning pre-trained models: 1. Load a pre-trained model with Dense/FullyConnected layers 2. Wrap those layers with DenseLoRAAdapter 3. Freeze the base layers 4. Train only the small LoRA corrections 5. Achieve similar results with 100x fewer trainable parameters! Example: If you have a dense layer with 1000x1000 weights, wrapping it with rank=8 LoRA (frozen) reduces trainable parameters from 1,000,000 to just 16,000!

## How It Works

The DenseLoRAAdapter wraps Dense or FullyConnected layers and adds a LoRA layer in parallel. During forward pass, both the base layer and LoRA layer process the input, and their outputs are summed. The base layer's parameters can be frozen while only the LoRA parameters are trained.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new DenseLoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured DenseLoRAAdapter (rank {config.Rank}).");
```

