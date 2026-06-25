---
title: "QLoRAAdapter"
description: "QLoRA (Quantized LoRA) adapter for parameter-efficient fine-tuning with 4-bit quantized base weights."
section: "Reference"
---

_LoRA / PEFT Adapters_

QLoRA (Quantized LoRA) adapter for parameter-efficient fine-tuning with 4-bit quantized base weights.

## For Beginners

QLoRA is an advanced technique that makes fine-tuning large models even more memory-efficient than standard LoRA. Here's how it works: Imagine you have a huge model with millions of parameters: - Standard LoRA: Freezes the base model, trains small adapters (huge memory savings) - QLoRA: Does the same BUT also compresses the base model to 4-bit (even more savings!) Think of it like storing a high-resolution image: - Original model: Full 16-bit floating point (2 bytes per number) - QLoRA base: Compressed to 4-bit (0.5 bytes per number) - LoRA adapters: Still full precision (for accurate learning) The result: You can fine-tune models 4x larger on the same hardware, or use 4x less GPU memory! **When to use QLoRA vs Standard LoRA:** - Use QLoRA when: GPU memory is very limited, model is huge, inference speed is critical - Use Standard LoRA when: Memory is not a constraint, maximum accuracy is needed - Both achieve similar quality in practice, QLoRA just uses less memory **Trade-offs:** - Pros: 75% less memory, same performance as 16-bit LoRA, faster inference after merging - Cons: Slightly slower forward pass (dequantization overhead), more complex implementation

## How It Works

QLoRA extends the LoRA (Low-Rank Adaptation) technique by quantizing the base layer's weights to 4-bit precision while keeping the LoRA adapter matrices (A and B) in full precision. This achieves dramatic memory savings (typically 4x reduction) while maintaining training quality comparable to full 16-bit fine-tuning. 

**Key Features:** - Base layer weights stored in 4-bit precision (INT4 or NF4) - LoRA matrices (A and B) remain in full precision for accurate gradient updates - Double quantization for constant quantization parameters (further memory savings) - Paged optimizers support for handling memory spikes during training - Dequantization happens on-the-fly during forward pass 

**Memory Savings:** For a typical transformer layer with 1000x1000 weights: - Standard 16-bit: 2MB for weights - QLoRA 4-bit base: 0.5MB for base weights + full precision LoRA (e.g., 32KB for rank 8) - Total savings: ~75% memory reduction on base weights 

**Quantization Types:** - INT4: Uniform 4-bit integer quantization (-8 to 7) - NF4 (4-bit Normal Float): Information-theoretically optimal for normally distributed weights 

**Research Background:** QLoRA was introduced in "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023). It enables fine-tuning of 65B parameter models on a single 48GB GPU by combining: 1. 4-bit NormalFloat (NF4) quantization optimized for normally distributed weights 2. Double quantization to reduce memory footprint of quantization constants 3. Paged optimizers to handle memory spikes during gradient checkpointing

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new QLoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured QLoRAAdapter (rank {config.Rank}).");
```

